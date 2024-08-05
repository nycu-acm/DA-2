import logging
import torch
import torch.nn.functional as F
import numpy as np
from ..loss import build_criterion_from_cfg
from ..loss import SWD

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss_raw = -(one_hot * log_prb).sum(dim=1)
        loss = loss_raw.mean()

    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss, loss_raw



def get_dada_augloss(cfg, model_student, model_teacher, data_real, data_fake, epoch, summary, writer):  
    #   get loss on real/fake data
    model_student.eval()
    pred_fake_stu = model_student.forward(data_fake)                     #   [B, 40]
    pred_real_stu = model_student.forward(data_real)                     #   [B, 40]
    label = data_real['y']
    criterion_stu = build_criterion_from_cfg(cfg.student_criterion_args)
    loss_fake_stu = criterion_stu(pred_fake_stu, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    
    #   Generalization Loss
    g_loss = torch.sum(torch.square(loss_fake_stu))

    #   Teacher-Student Loss
    # model_teacher.eval()
    pred_fake_tea = model_teacher.forward(data_fake)                     #   [B, 40]
    criterion_tea = build_criterion_from_cfg(cfg.teacher_criterion_args)
    loss_fake_tea = criterion_tea(pred_fake_tea, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    # loss_real_tea = criterion_tea(pred_real_tea, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    teacher_loss = torch.abs(1 - torch.exp(loss_fake_stu - cfg.loss_weight.w_tea * loss_fake_tea))

    #   SWD Loss
    real_points = data_real['x']
    fake_points = data_fake['x']
    # print(f'original_point_num{len(real_points)}, augmented_point_num{len(fake_points)}')
    SWD_criterion = SWD(num_projs=100)
    SWD_loss_dict = SWD_criterion(real_points, fake_points)

    swd_loss = SWD_loss_dict['loss'].mean(dim=0)

    w_swd = cfg.loss_weight.w_swd
    w_tea_stu = cfg.loss_weight.w_tea_stu
    w_gloss = cfg.loss_weight.w_gloss
    feedback_loss = w_swd*swd_loss + w_tea_stu*teacher_loss + w_gloss*g_loss

    writer.add_scalar('train_G_iter/loss_fake_stu', loss_fake_stu.mean().item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_fake_tea', loss_fake_tea.mean().item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/g_loss', g_loss.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/swd_loss', swd_loss.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/teacher_loss', teacher_loss.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/feedback_loss', feedback_loss.item(), summary.train_iter_num)

    return feedback_loss, pred_fake_stu
