#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 13:56
# @Author  : wangjie

import __init__
import os, argparse, yaml, numpy as np
from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
import open3d as o3d
import torch
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from tsnecuda import TSNE
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import sklearn
from sklearn.utils import resample
import random

color_map = {
    0: 'blue',
    7: 'orange',
    8: 'green',
    12: 'red',
    17: 'brown',
    19: 'purple',
    20: 'pink',
    23: 'gray',
    24: 'olive',
    25: 'cyan',
    26: 'magenta',
}
[26, 23, 8, 25, 19]
shape_list = {
    0: '+',
    7: '^',
    8: 'o',#
    12: '2',
    17: '1',
    19: 'D',#
    20: '3',
    23: '^',#
    24: '4',
    25: 's',#
    26: '*',#
}
class_list = {
    0: 'airplane',
    7: 'car',
    8: 'chair',
    12: 'desk',
    17: 'guitar',
    19: 'lamp',
    20: 'laptop',
    23: 'night_stand',
    24: 'person',
    25: 'piano',
    26: 'plant',
}

def hsv_to_rgb(hsv):
    """Convert HSV color to RGB color"""
    h, s, v = hsv
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return np.array([r + m, g + m, b + m])

class HDF5Dataset(Dataset):
    def __init__(self, file_path, filter_labels):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.pointcloud = f['pointcloud'][:]
            self.raw = f['raw'][:]
            self.label = f['label'][:]
            self.accuaug_cnt = f['accuaug_cnt'][:]
            self.index = f['index'][:]
        
        if filter_labels is not None:
            self.filter_data_by_labels(filter_labels)

    def filter_data_by_labels(self, filter_labels):
        # Create a boolean mask where any of the labels match
        mask = np.isin(self.label, filter_labels)
        self.pointcloud = self.pointcloud[mask]
        self.raw = self.raw[mask]
        self.label = self.label[mask]
        self.accuaug_cnt = self.accuaug_cnt[mask]
        self.index = self.index[mask]

    def __len__(self):
        return self.pointcloud.shape[0]

    def __getitem__(self, idx):
        pointcloud = self.pointcloud[idx]
        raw = self.raw[idx]
        label = self.label[idx]
        accuaug_cnt = self.accuaug_cnt[idx]
        index = self.index[idx]
        data = {
            'pointcloud': pointcloud,
            'raw': raw,
            'label': label,
            'accuaug_cnt': accuaug_cnt,
            'index': index,
        }
        return data

def load_h5(h5_name, filter_labels, item):
    f = h5py.File(h5_name, 'r')
    pointcloud = f['pointcloud'][:].astype('float32')
    label = f['label'][:].astype('int64')
    accuaug_cnt = f['accuaug_cnt'][:].astype('int64')
    before_masked = f['before_masked'][:].astype('float32')
    
    
    mask = np.isin(label, filter_labels)
    pointcloud = pointcloud[mask]
    label = label[mask]
    accuaug_cnt = accuaug_cnt[mask]
    before_masked = before_masked[mask]

    pointcloud = pointcloud[item]
    label = label[item]
    accuaug_cnt = accuaug_cnt[item]
    before_masked = before_masked[item]
    data = {
        'pointcloud': pointcloud,
        'label': label,
        'accuaug_cnt': accuaug_cnt,
        'before_masked': before_masked,
    }
    f.close()
    return data

def o3dvis_multi(pointclouds_list, auglevel_list=None, sample_color=None):
    '''
        points_1: coors of feat,[npoint, 3]
        points_2: coors of feat,[npoint, 3]
    '''
    width = 1080
    height = 1080

    front_size = [-0.078881283843093356, 0.98265290049739873, -0.16784224796908237]
    lookat_size = [0.057118464194894254, -0.010695673712330742, 0.047245129152854129]
    up_size = [0.018854223129214469, -0.16686616421723113, -0.98579927039414161]
    zoom = 0.98
    front = np.array(front_size)
    lookat = np.array(lookat_size)
    up = np.array(up_size)
    
    # R_x = o3d.geometry.get_rotation_matrix_from_axis_angle((0, 0, 0))
    # R_y = o3d.geometry.get_rotation_matrix_from_axis_angle((0, np.pi/6, 0))  # 30 degrees in radians
    # R_z = o3d.geometry.get_rotation_matrix_from_axis_angle((0, 0, np.pi))

    R_x = o3d.geometry.get_rotation_matrix_from_axis_angle((0, 0, 0))
    R_y = o3d.geometry.get_rotation_matrix_from_axis_angle((0, 0, 0))  # 30 degrees in radians
    R_z = o3d.geometry.get_rotation_matrix_from_axis_angle((0, 0, 0))

    # Combine the rotation matrices to get the total transformation matrix
    T = np.dot(R_x, R_y)
    T = np.dot(T, R_z)
    T = np.pad(T, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    T[3, 3] = 1

    # print("Transformation Matrix:")
    # print(T)

    chessboard_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0.1, 0.1, 0.1])
    
    P_list = []

    for i in range(len(pointclouds_list)):
        points = pointclouds_list[i]

        if auglevel_list is not None:
            aug_level = auglevel_list[i]
            
            if sample_color is not None:
                rgb = sample_color[i]
            else:
                hue = 200 + (0.1*aug_level) * 160
                value = 0.9
                sat = 1
                hsv = np.array([hue, sat, value])
                rgb = hsv_to_rgb(hsv)
        else:
            hsv = np.array([200, 1, 1])
            rgb = hsv_to_rgb(hsv)     

        points = points - [1, 0, 0]
        color = np.zeros_like(points)
        
        color[:] = rgb
        # color_new[:] = green_rgb
        P = o3d.geometry.PointCloud()
        P.points = o3d.utility.Vector3dVector(points)
        P.colors = o3d.utility.Vector3dVector(color)
        P.transform(T)
        P.translate([i*2.5, 0, 0])
        P_list.append(P)
   
    # o3d.visualization.draw_geometries(P_list, zoom=zoom, front=front, lookat=lookat, up=up)
    
    o3d.visualization.draw_geometries(P_list)

def undersample_classes(X, y):
    """
    Undersample each class in the dataset to match the size of the smallest class.
    
    Parameters:
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
    y (numpy.ndarray): Label vector of shape (n_samples,).
    
    Returns:
    X_balanced (numpy.ndarray): Balanced feature matrix.
    y_balanced (numpy.ndarray): Balanced label vector.
    """
    # Determine the number of samples in the minority class
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_size = class_counts.min()
    
    # Initialize lists to hold the undersampled data
    X_balanced_list = []
    y_balanced_list = []
    # Undersample each class to the size of the minority class
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        np.random.seed(0)
        undersample_indices = np.random.choice(class_indices, size=min_class_size, replace=False)
        
        X_balanced_list.append(X[undersample_indices])
        y_balanced_list.append(y[undersample_indices])
    
    # Combine the undersampled data from all classes
    X_balanced = np.vstack(X_balanced_list)
    y_balanced = np.hstack(y_balanced_list)
    
    # Shuffle the balanced dataset
    shuffle_indices = np.random.permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]
    
    return X_balanced, y_balanced

def get_analyze_data_raw(file_path, model, filter_labels):
    # Create the dataset
    dataset = HDF5Dataset(file_path, filter_labels)
    # dataset_filtered = HDF5Dataset_filtered(file_path)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    model.eval()
    # feature PCA
    pbar = tqdm(enumerate(dataloader), total=dataloader.__len__())
    features_raw = None
    labels = None

    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        origin_points = data['raw']
        label = data['label']

        data_raw = {
            'pos': origin_points[:, :, :3].contiguous(),
            'y': label,
            'x': origin_points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }

        feat_real = model(data_raw)
        current_feature_raw = feat_real.detach().cpu().numpy()
        current_label = label.detach().cpu().numpy()
        # print(current_feature_fake.shape)
        if features_raw is not None:
            features_raw = np.concatenate((features_raw, current_feature_raw))
            labels = np.concatenate((labels, current_label))
        else:
            features_raw = current_feature_raw
            labels = current_label

    analyze_data_raw = [features_raw, labels]
    return analyze_data_raw

def get_viz_data(root_path, all_files, label_list, item_list, epoch_list, epoch_range=None):
    data_path = f"{root_path}/fakedata"
    vis_points_list = []
    vis_augcnt = []
    
    for i in range(len(item_list)):
        item = item_list[i]
        label = label_list[i]
        if epoch_range is not None:
            selected_files = [f for f in all_files if (epoch_range[0] <= int(pattern.match(f).group(1)) <= epoch_range[1])]
        else:
            if len(epoch_list[i]) > 2:
                selected_files = [f for f in epoch_files if int(pattern.match(f).group(1)) in epoch_list[i]]
            else:
                selected_files = [f for f in all_files if (epoch_list[i][0] <= int(pattern.match(f).group(1)) <= epoch_list[i][1])]

        accuaug_cnt_list = []
        label_aug_list = []
        for count, file_name in enumerate(selected_files):
            file_path = os.path.join(data_path, file_name)
            data = load_h5(file_path, label, item)
            points = data['pointcloud']
            accuaug_cnt = data['accuaug_cnt']
            label_aug = data['label']

            if accuaug_cnt == 0:
                # augmented_points.append(before_masked)
                label_aug_list.append(label_aug)
                accuaug_cnt_list.append(int(-1))

            vis_points_list.append(points)
            vis_augcnt.append(accuaug_cnt)

            # augmented_points.append(points)
            label_aug_list.append(label_aug)
            accuaug_cnt_list.append(accuaug_cnt)

        print(class_list[label], accuaug_cnt_list)

    return vis_points_list, vis_augcnt

def get_analyze_data(root_path, all_files, model, filter_labels, label_list, item_list, epoch_list, epoch_range=None):
    data_path = f"{root_path}/fakedata"
    analyze_augdata_list = []
    vis_points_list = []
    vis_augcnt = []
    
    for i in range(len(item_list)):
        item = item_list[i]
        label = label_list[i]
        if epoch_range is not None:
            selected_files = [f for f in all_files if (epoch_range[0] <= int(pattern.match(f).group(1)) <= epoch_range[1])]
        else:
            if len(epoch_list[i]) > 2:
                selected_files = [f for f in epoch_files if int(pattern.match(f).group(1)) in epoch_list[i]]
            else:
                selected_files = [f for f in all_files if (epoch_list[i][0] <= int(pattern.match(f).group(1)) <= epoch_list[i][1])]

        augmented_points = []
        accuaug_cnt_list = []
        label_aug_list = []
        for count, file_name in enumerate(selected_files):
            file_path = os.path.join(data_path, file_name)
            # Create the dataset
            # dataset = HDF5Dataset(file_path, filter_labels[l])
            data = load_h5(file_path, label, item)
            # data = dataset[item]
            points = data['pointcloud']
            accuaug_cnt = data['accuaug_cnt']
            label_aug = data['label']
            before_masked = data['before_masked']

            if accuaug_cnt == 0:
                augmented_points.append(before_masked)
                label_aug_list.append(label_aug)
                accuaug_cnt_list.append(int(-1))

            vis_points_list.append(points)
            vis_augcnt.append(accuaug_cnt)

            augmented_points.append(points)
            label_aug_list.append(label_aug)
            accuaug_cnt_list.append(accuaug_cnt)

        print(class_list[label], accuaug_cnt_list)
        augmented_points = torch.tensor(np.stack(augmented_points))
        augmented_points = augmented_points.to('cuda')
        # print(augmented_points.shape)
        data_aug = {
                'pos': augmented_points[:, :, :3],
                'x': augmented_points[:, :, :cfg.model.in_channels].transpose(1, 2),
            }
        feat_aug = model(data_aug)
        feat_aug = feat_aug.detach().cpu().numpy()
        augmented_points_list = augmented_points.detach().cpu().numpy()

        analyze_data_aug = [feat_aug, accuaug_cnt_list, label_aug_list, augmented_points_list]
    
        analyze_augdata_list.append(analyze_data_aug)

    return analyze_augdata_list, vis_points_list, vis_augcnt

def pca_viz(analyze_data_raw, analyze_augdata_list, epoch_range, label_list, item_list, pca_class_path):
    features_raw, label_raw_list = analyze_data_raw
    features_raw, label_raw_list = undersample_classes(features_raw, label_raw_list)
    
    feat_aug_list = [data[0] for data in analyze_augdata_list]
    shapes = [len(feat) for feat in feat_aug_list]
    features_aug = np.concatenate(feat_aug_list, axis=0)

    accuaug_cnt_list = np.concatenate([data[1] for data in analyze_augdata_list], axis=0)
    label_aug_list = np.concatenate([data[2] for data in analyze_augdata_list], axis=0)
    augmented_points_list = np.concatenate([data[3] for data in analyze_augdata_list], axis=0)

    concatenated_features = np.concatenate((features_raw, features_aug), axis=0)
    scaler = sklearn.preprocessing.RobustScaler()
    # concatenated_features = scaler.fit_transform(concatenated_features)
    pca_points = PCA(n_components=2).fit_transform(concatenated_features)
    all_label = np.concatenate((label_raw_list, label_aug_list))
    # all_augcnt = np.concatenate((np.zeros_like(label_raw_list), accuaug_cnt_list))
    pca_points[all_label == 26, 1] += 1
    
    # scaler = sklearn.preprocessing.RobustScaler()
    # pca_points = scaler.fit_transform(pca_points)

    # scaler = sklearn.preprocessing.PowerTransformer(method='yeo-johnson', standardize=False)
    # scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='uniform')
    # pca_points_normalized_x = pca_points.copy()
    # pca_points[:, 1] = scaler.fit_transform(pca_points[:, [1]]).flatten()
    pca_raw_points = pca_points[:len(features_raw)]
    pca_aug_points = pca_points[len(features_raw):]

    # Step 3: Split the concatenated array back to original segments
    split_indices = np.cumsum(shapes)[:-1]  # Calculate split indices
    pca_aug_split = np.split(pca_aug_points, split_indices)
    # print(pca_aug_split)

    plt.figure(dpi=800)
    s = 7
    alpha = 0.5
    sample_color = []
    # Create a figure and axis
    viz_point_list = []
    # Plot original samples
    for label in label_list:
        idx = np.where(label_raw_list == label)
        
        plt.scatter(pca_raw_points[idx, 0], pca_raw_points[idx, 1], s=s, marker=shape_list[label], label=class_list[label], color='grey', alpha=alpha)
        # plt.scatter(pca_raw_points[idx, 0], pca_raw_points[idx, 1], s=7, marker='o', label=class_list[label], color=color_map[label], alpha=0.5)
    
    # Plot augmented samples
    for i in range(len(analyze_augdata_list)):
        pca_aug_points_class = pca_aug_split[i]
        augcnt = analyze_augdata_list[i][1]
        label = analyze_augdata_list[i][2]
        
        for j in range(len(pca_aug_points_class)):
            # print(pca_aug_points_class[j])
            if augcnt[j] >= 0:
                hue = 200 + (0.1*augcnt[j]) * 160
                value = 0.9
                sat = 1
            else:
                hue = 360
                value = 1
                sat = 1
            
            hsv = np.array([hue, sat, value])
            rgb = hsv_to_rgb(hsv)
            sample_color.append(rgb)
            if augcnt[j] == -1:

                if j > 0:
                    plt.scatter(pca_aug_points_class[j][0], pca_aug_points_class[j][1], s=s, color='black', alpha=1, marker='x')
            else:
                plt.scatter(pca_aug_points_class[j][0], pca_aug_points_class[j][1], s=s, color=rgb, alpha=alpha, marker=shape_list[label[j]])
                # plt.scatter(pca_aug_points_class[j][0], pca_aug_points_class[j][1], s=5+augcnt[j]*0.5, color=rgb, alpha=0.5, marker='o')

            # if j >= 1:
            #     if augcnt[j-1] < augcnt[j]:
            #         if augcnt[j] == 0:
            #             arrow = FancyArrowPatch((pca_aug_points_class[j-1][0], pca_aug_points_class[j-1][1]), (pca_aug_points_class[j][0], pca_aug_points_class[j][1]), 
            #                             connectionstyle="arc3,rad=.3", color='grey', arrowstyle='->', mutation_scale=15, lw=0.2, ls='--')
            #             plt.gca().add_patch(arrow)
            #         else:
            #             plt.arrow(pca_aug_points_class[j-1][0], pca_aug_points_class[j-1][1], pca_aug_points_class[j][0]-pca_aug_points_class[j-1][0], pca_aug_points_class[j][1]-pca_aug_points_class[j-1][1], 
            #                     color='black', linewidth=0.2, length_includes_head=True)
            #     elif augcnt[j-1] > augcnt[j]:
            #         plt.arrow(pca_aug_points_class[j-1][0], pca_aug_points_class[j-1][1], pca_aug_points_class[j][0]-pca_aug_points_class[j-1][0], pca_aug_points_class[j][1]-pca_aug_points_class[j-1][1], 
            #                 color='red', linewidth=0.2, length_includes_head=True)
                               
    plt.legend()
    plt.savefig(f'{pca_class_path}/{label_list}_{item_list}.png')
    # plt.savefig(f'{pca_class_path}/{item_list[0]}.png')
    plt.clf()
    o3dvis_multi(augmented_points_list, accuaug_cnt_list, sample_color)

def find_checkpoint_file(folder_path, keyword='ckpt_best.pth'):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if keyword in file:
                return os.path.join(root, file)
    return None

def sort_by_number(filename):
    return int(filename.split('_')[1].split('.')[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--l', type=int, default=0, help='label')
    parser.add_argument('--item', type=int, default=0, help='item')
    # parser.add_argument('--msg_ckpt', default='ngpus1-seed6687-20230208-113827-C5JvDVLy53nEazUYXPtNyj', type=str, help='message after checkpoint')
    # parser.add_argument('--root_path', default=None, type=str, help='message after checkpoint')
    

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    torch.cuda.empty_cache()
    #   loading model
    model = build_model_from_cfg(cfg.model).cuda()
    # print(model)

    #   loading ckpt dir
    # root_path = args.root_path
    # root_path = f"/mnt/HDD8/max/TeachAugment_point/log/modelnet_dynamic_threshold/teachaugpoint_weight-pointnext-20240522-143251-Dycdfmrveg72cNLNECFDri"
    # root_path = f"/mnt/HDD8/max/TeachAugment_point/log/modelnet_dynamic_threshold/teachaugpoint_weight-pointnext-20240618-161406-izURqJ6jcd7kj4tb9ypnBq"
    # root_path = f"/mnt/HDD8/max/TeachAugment_point/log/modelnet_dynamic_threshold/teachaugpoint_weight-pointnext-20240626-175944-kf6dxoaEZSK3HK6ei9SSzT"
    root_path = f"/mnt/HDD8/max/TeachAugment_point/log/modelnet_dynamic_threshold/teachaugpoint_weight-pointnext-20240627-230506-BqhcRCgTLspsewe253r38K"
    checkpoint_path = f'{root_path}/checkpoint'
    ckpt_file_path = find_checkpoint_file(checkpoint_path)

    #   loading state dict
    state_dict = torch.load(ckpt_file_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])

    data_path = f"{root_path}/fakedata"

    # PCA_path = f"{root_path}/PCA"
    # if not os.path.isdir(PCA_path):
    #     os.makedirs(PCA_path)
    
    # TSNE_path = f"{root_path}/TSNE"
    # if not os.path.isdir(TSNE_path):
    #     os.makedirs(TSNE_path)

    all_files = os.listdir(data_path)
    pattern = re.compile(r'^epoch(\d+)\.h5$')
    epoch_files = sorted([f for f in all_files if pattern.match(f)], key=lambda x: int(pattern.match(x).group(1)))

    
    # filter_labels = [0, 7, 8, 12, 17, 19, 20, 23, 24, 25, 26]
    class_list = {
        0: 'airplane',
        7: 'car',
        8: 'chair',
        12: 'desk',
        17: 'guitar',
        19: 'lamp',
        20: 'laptop',
        23: 'night_stand',
        24: 'person',
        25: 'piano',
        26: 'plant',
    }
    
    # label = 8
    # item = 38
    # epochs = [146, 150, 151, 152, 153, 154, 157, 158, 160, 162] # chair 38
    # label_list.append(label)
    # item_list.append(item)
    # epoch_list.append(epochs)
   
    # epochs = [108, 117]
    # filter_labels = [26, 23, 8, 25, 19]

    # label_list = filter_labels
    # item_list = [12, 23, 7, 23, 0]
    # epoch_list = [[118, 127], [121, 130], [108, 117], [111, 120], [115, 124]]

   
    # [0, 7, 8, 12, 17, 19, 20, 23, 24, 25, 26]
    
    # filter_labels = [random.choice([0])]
    item_list = [random.randrange(150)]
    # e = random.randrange(51, 150)
    # epoch_list = [[e, e+15]]

    filter_labels = [12]
    # item_list = [16]
    epoch_list = [[101, 125]]

    label_list = filter_labels
    print(item_list[0], epoch_list[0])
    # label_list = []
    # item_list = []
    # epoch_list = []
    # for i in range(20):
    #     item = i
    #     label_list.append(filter_labels[0])
    #     item_list.append(item)
    #     epoch_list.append(epochs)
    
    # pca_class_path = f"{root_path}/PCA/multi_{epoch_list[0]}/{class_list[label_list[0]]}"
    # if not os.path.isdir(pca_class_path):
    #     os.makedirs(pca_class_path)


    # file_path = os.path.join(data_path, epoch_files[0])
    # analyze_data_raw = get_analyze_data_raw(file_path, model, filter_labels)
    # analyze_augdata_list, vis_points_list, vis_augcnt = get_analyze_data(root_path, epoch_files, model, filter_labels, label_list, item_list, epoch_list)
    # pca_viz(analyze_data_raw, analyze_augdata_list, epochs, filter_labels, item_list, pca_class_path)

    vis_points_list, vis_augcnt = get_viz_data(root_path, epoch_files, label_list, item_list, epoch_list)
    o3dvis_multi(vis_points_list, vis_augcnt)

    # epochs = [111, 130]
    # for label in filter_labels:
    #     pca_class_path = f"{root_path}/PCA/multi_{epochs}/{class_list[label]}"
    #     if not os.path.isdir(pca_class_path):
    #         os.makedirs(pca_class_path)
    #     for i in range(30):
    #         print(class_list[label]) 
    #         item = i
    #         epoch_list = []
    #         label_list = []
    #         item_list = []
    #         label_list.append(label)
    #         item_list.append(item)
    #         epoch_list.append(epochs)

    #         analyze_augdata_list = get_analyze_data(root_path, epoch_files, model, filter_labels, label_list, item_list, epoch_list, epoch_range=epochs)
    #         pca_viz(analyze_data_raw, analyze_augdata_list, epochs, filter_labels, item_list, pca_class_path)

