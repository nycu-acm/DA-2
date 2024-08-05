#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 16:35
# @Author  : wangjie

import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious, PCA
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from examples.classification.train_pointwolf_utils import train_one_epoch_pointwolf, train_one_epoch_rsmix
from openpoints.dada_models import build_dadamodels_from_cfg
from openpoints.function_dada import Form_dataset_cls, get_dada_augloss
from openpoints.online_aug.pointwolf import PointWOLF_classversion
import h5py
from openpoints.utils import Summary

import matplotlib.pyplot as plt

def copyfiles(cfg):
    import shutil
    #   copy pointcloud model
    path_copy = f'{cfg.run_dir}/copyfile'
    if not os.path.isdir(path_copy):
        os.makedirs(path_copy)
    shutil.copy(f'{os.path.realpath(__file__)}', path_copy)
    shutil.copytree('openpoints', f'{path_copy}/openpoints')
    pass

def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()

def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)

def save_augmenter(generator, path, epoch):
    state = {
        'generator': generator.state_dict(),
    }
    path = path + '/augmenter'
    if not os.path.isdir(path):
        os.makedirs(path)
    filepath = os.path.join(path, f"augmenter_{epoch}.pth")
    torch.save(state, filepath)

def get_gan_model(cfg):
    print("==> Creating model...")
    # generator
    generator = build_dadamodels_from_cfg(cfg.dada_augmenter).cuda()
    # generator = PointWOLF_classversion4().to(device)
    print("==> Total parameters of Generator: {:.2f}M"\
          .format(sum(p.numel() for p in generator.parameters()) / 1000000.0))

    if cfg.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.optim_args.lr_generator, betas=(cfg.optim_args.b1, cfg.optim_args.b2))

    criterion_gan = torch.nn.BCELoss()
    dict = {
        'model_G': generator,
        'optimizer_G': optimizer_G,
        'criterion_gan': criterion_gan
    }
    return dict

def train_gan(cfg, gan_dict, train_loader, summary, writer, epoch, model_student, model_teacher):
    generator = gan_dict['model_G']
    optimizer_G = gan_dict['optimizer_G']
    criterion_gan = gan_dict['criterion_gan']
    generator.train()
    model_student.eval()
    model_teacher.eval()
    # pointwolf = PointWOLF_classversion(**cfg.pointwolf)
    for i, data in tqdm(enumerate(train_loader), total=train_loader.__len__()):
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        if 'origin_x' in data and data['origin_x'] is not None:
            origin_x = data['origin_x']
        else:
            origin_x = data['x'].clone()
        points = data['x']
        label = data['y']
        idx = data['idx']
        if 'unmasked_pos' in data and data['unmasked_pos'] is not None:
            points[:, :, :3] = data['unmasked_pos']
        points_clone = points.clone()
        # points_unmasked = points.clone()
        input_pointcloud = points[:, :, :3].contiguous()

        #  Train Generator
        _, _, gen_imgs = generator(input_pointcloud)
 

        points[:, :, :3] = gen_imgs

        data_fake = {
            'pos': points[:, :, :3].contiguous(),
            'y': label,
            'x': points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }
        data_real = {
            'pos': origin_x[:, :, :3].contiguous(),
            'y': label,
            'x': origin_x[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }

        feedback_loss, _ = get_dada_augloss(cfg=cfg, model_student=model_student, model_teacher=model_teacher, \
                                            data_real=data_real, data_fake=data_fake, \
                                            epoch=epoch, summary=summary, writer=writer)
            
        optimizer_G.zero_grad()
        feedback_loss.backward(torch.ones_like(feedback_loss))

        optimizer_G.step()
        
    model_student.zero_grad()


def gen_augdata(cfg, gan_dict, train_loader, epoch, model, avg_class_score, summary, writer):
    generator = gan_dict['model_G']
    generator.eval()
    model.eval()
    # prepare buffer list for update
    tmp_out_buffer_list = []
    tmp_points_buffer_list = []
    tmp_label_buffer_list = []
    tmp_idx_buffer_list = []
    tmp_unmasked_buffer_list = []
    tmp_originx_buffer_list = []
    tmp_accu_aug_count_list = []
    tmp_points_before_masked_buffer_list = []
    aug_count = 0
    for i, data in tqdm(enumerate(train_loader), total=train_loader.__len__()):
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        if 'origin_x' in data and data['origin_x'] is not None:
            origin_x = data['origin_x']
        else:
            origin_x = data['x'].clone()
        points = data['x']
        label = data['y']
        if 'accuaug_cnt' in data and data['accuaug_cnt'] is not None:
            accuaug_cnt = data['accuaug_cnt']
        else:
            accuaug_cnt = torch.zeros_like(label)
        index = data['idx']
        if 'unmasked_pos' in data and data['unmasked_pos'] is not None:
            points[:, :, :3] = data['unmasked_pos']

        input_pointcloud = points[:, :, :3].contiguous()

        _, unmasked_pos, gen_imgs = generator(input_pointcloud)

        points[:, :, :3] = gen_imgs

        data_fake = {
            'pos': points[:, :, :3].contiguous(),
            'y': label,
            'x': points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }

        pred_fake_stu = model.forward(data_fake)

        pred_fake_stu = torch.nn.functional.softmax(pred_fake_stu, dim=1)

        if cfg.use_CDTS is True:
            out_points, unmasked_pos, count, accuaug_cnt, points_before_masked = mask_data(cfg, pred_fake_stu, points, unmasked_pos, origin_x, label, epoch, avg_class_score, accuaug_cnt)
            aug_count += count
            tmp_points_before_masked_buffer_list.append(points_before_masked.detach().cpu().numpy())
        else:
            out_points = points
            # filtered_points = None      

        tmp_out_buffer_list.append(out_points[:, :, :3].detach().cpu().numpy())
        tmp_label_buffer_list.append(label.detach().cpu().numpy())
        tmp_points_buffer_list.append(out_points.detach().cpu().numpy())
        tmp_unmasked_buffer_list.append(unmasked_pos.detach().cpu().numpy())
        tmp_originx_buffer_list.append(origin_x.detach().cpu().numpy())
        tmp_accu_aug_count_list.append(accuaug_cnt.detach().cpu().numpy())
        tmp_idx_buffer_list.append(index.detach().cpu().numpy())
        
    # aug_level, counts = torch.unique(torch.Tensor(tmp_accu_aug_count_list).to(torch.int), return_counts=True)

    writer.add_scalar('train_G_iter/aug_count', aug_count, summary.train_iter_num)
    data_transform = build_transforms_from_cfg('train', cfg.datatransforms)
    fake_dataset = Form_dataset_cls(tmp_out_buffer_list, tmp_label_buffer_list, tmp_points_buffer_list, tmp_unmasked_buffer_list, tmp_originx_buffer_list, tmp_accu_aug_count_list, tmp_idx_buffer_list, data_transform)


    return fake_dataset

def mask_data(cfg, pred_fake_stu, points, unmasked_pos, origin_x, label, epoch, avg_class_score, accuaug_cnt):
    device = label.device
    score_stu = torch.gather(pred_fake_stu, 1, label.unsqueeze(1))
    score_stu = torch.flatten(score_stu)
    
    avg_class_score = avg_class_score.to(device)

    avg_score = torch.zeros_like(label, device=device, dtype=torch.float)
    for i in range(len(label)):
        l = label[i]
        avg_score[i] = avg_class_score[l]

    th = get_threshold(cfg, epoch, avg_score)

    mask_cnt = accuaug_cnt < cfg.reset_count
    mask_score = score_stu > th
    mask = mask_cnt * mask_score

    accuaug_cnt = accuaug_cnt * mask + mask
    count = mask.sum()
    
    mask = mask.unsqueeze(-1).unsqueeze(-1)   

    # mask out samples with low confidence, reset to original input
    points_before_masked = points.clone()
    points = torch.where(mask, points, origin_x)
    unmasked_pos = torch.where(mask, unmasked_pos, origin_x[:, :, :3])

    
    return points, unmasked_pos, count, accuaug_cnt, points_before_masked

def get_threshold(cfg, epoch, avg_score):
    # Selection Stage:    
    if epoch > cfg.ema_args.ema_warmup_epochs:
        t = epoch - cfg.ema_args.ema_warmup_epochs + 1
        gamma = cfg.threshold_decay_rate

        threshold = (pow(gamma, -t) * (cfg.threshold_start - cfg.lower_threshold) + cfg.lower_threshold) * avg_score
    else:
        threshold = 500.0
    return threshold


def train_one_epoch(model, teacher_model, train_loader, optimizer, scheduler, epoch, cfg, summary, writer):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    model.train()  # set model to training mode
    teacher_model.train()

    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    scores_by_class = torch.zeros(cfg.num_classes)
    num_by_class = torch.zeros(cfg.num_classes, dtype=torch.int)
    for idx, data in pbar:
        # print(data.keys())
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        points = data['x']
        target = data['y']
        index = data['idx']
        num_curr_pts = points.shape[1]
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(
                points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]
            points = torch.gather(
                points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

        logits, loss = model.get_logits_loss(data, target) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target) 

        loss.backward()


        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)
        
        # update ema
        if cfg.pretrained_teacher.use_pretrained is False:
            if idx % 32 == 0:
                teacher_model.update_parameters(model)
                if epoch < cfg.ema_args.ema_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    teacher_model.n_averaged.fill_(0)
            
        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
        
        # get scores
        # print(target, target.size())
        logits = logits.detach().cpu()
        logits = torch.nn.functional.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        for i in range(len(target)):
            if pred[i] == target[i]:
                c = target[i]
                scores_by_class[c] += logits[i][c]
                num_by_class[c] += 1
        # print(scores_by_class)
        # print(num_by_class)

    macc, overallacc, accs = cm.all_acc()
    avg_class_score = scores_by_class.div(num_by_class)
    print(avg_class_score)
    avg_score = scores_by_class.sum()/num_by_class.sum()
    writer.add_scalar('train_G_iter/avg_score', avg_score, summary.train_iter_num)
    summary.summary_train_iter_num_update()
    # print(avg_class_score)
    return loss_meter.avg, macc, overallacc, accs, cm, avg_class_score
 


def main(gpu, cfg, profile=False):
    copyfiles(cfg)
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        # writer = SummaryWriter(log_dir=cfg.run_dir)
        summary = Summary(cfg.run_dir)
        writer = summary.create_summary()
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    ######################################################
    # pretrained teacher
    if cfg.pretrained_teacher.use_pretrained == True:
        teacher_model = build_model_from_cfg(cfg.teacher_model).cuda()
        logging.info('loading pretrained teacher model ...')
        teacher_state_dict = torch.load(cfg.pretrained_teacher.pretrained_teacher_path, map_location='cpu')
        teacher_model.load_state_dict(teacher_state_dict['model'])
    # EMA Teacher
    else:
        teacher_model = ExponentialMovingAverage(model, device=cfg.rank, decay=cfg.ema_args.ema_rate)


    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(
        val_loader.dataset, 'classes') else None or np.range(num_classes)
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    
    path_PCA = f'{cfg.run_dir}/PCA'
    if not os.path.isdir(path_PCA):
        os.makedirs(path_PCA)
    # ===> start training
    val_macc, val_oa, val_accs, best_val, macc_when_best, best_epoch = 0., 0., [], 0., 0., 0
    model.zero_grad()
    gan_model_dict = get_gan_model(cfg)
    avg_class_score = torch.ones(num_classes)
    
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1

        if epoch > cfg.ema_args.get('ema_warmup_epochs', 0):
            train_gan(cfg, gan_model_dict, train_loader, summary, writer, epoch, model, teacher_model)
            fake_dataset = gen_augdata(cfg, gan_model_dict, train_loader, epoch, model, avg_class_score, summary, writer)
            fake_train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                     cfg.dataset,
                                                     cfg.dataloader,
                                                     datatransforms_cfg=None,
                                                     split='train',
                                                     distributed=cfg.distributed,
                                                     dataset=fake_dataset,
                                                     )
            
            if cfg.use_DA2 is True:
                train_loader = fake_train_loader

                print('train fake')
                train_loss, train_macc, train_oa, _, _, avg_class_score = \
                    train_one_epoch(model, teacher_model, train_loader,
                                optimizer, scheduler, epoch, cfg, summary, writer)
            else:
                train_loss, train_macc, train_oa, _, _, avg_class_score = \
                    train_one_epoch(model, teacher_model, fake_train_loader,
                                optimizer, scheduler, epoch, cfg, summary, writer)
            
        else:
            print('warm up')
            train_loss, train_macc, train_oa, _, _, avg_class_score = \
                train_one_epoch(model, teacher_model, train_loader,
                                optimizer, scheduler, epoch, cfg, summary, writer)        

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_macc, val_oa, val_accs, val_cm = validate_fn(
                model, val_loader, cfg)
            tea_val_macc, tea_val_oa, tea_val_accs, _ = validate_fn(
                teacher_model, val_loader, cfg)
            is_best = val_oa > best_val
            if is_best:
                best_val = val_oa
                macc_when_best = val_macc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
            print_cls_results(val_oa, val_macc, val_accs, epoch, cfg)
            print_cls_results(tea_val_oa, tea_val_macc, tea_val_accs, epoch, cfg)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_oa {train_oa:.2f}, val_oa {val_oa:.2f}, best val oa {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_oa', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('mAcc_when_best', macc_when_best, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
        
        
    # test the last epoch
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, best_epoch)
        writer.add_scalar('test_macc', test_macc, best_epoch)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg) 

    if writer is not None:
        writer.close()
    if cfg.distributed:
        dist.destroy_process_group()


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        points = data['x']
        points = points[:, :npoints]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    tp, count = cm.tp, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return macc, overallacc, accs, cm


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)