#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/9 21:23
# @Author  : wangjie
import os
import glob
import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

o3d.visualization.webrtc_server.enable_webrtc()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../fake_data')

red_rgb = np.array([255 / 255., 107 / 255., 107 / 255.])
green_rgb = np.array([107 / 255., 203 / 255., 119 / 255.])
blue_rgb = np.array([77 / 255, 150 / 255, 255 / 255])
white = np.array([255 / 255, 255 / 255, 255 / 255])
purple = np.array([138 / 255, 163 / 255, 255 / 255])
red2green = red_rgb - green_rgb
green2blue = green_rgb - blue_rgb
red2blue = red_rgb - blue_rgb

# ScanObjectNN
# classes = [
#     "bag",
#     "bin",
#     "box",
#     "cabinet",
#     "chair",
#     "desk",
#     "display",
#     "door",
#     "shelf",
#     "table",
#     "bed",
#     "pillow",
#     "sink",
#     "sofa",
#     "toilet",
# ]

# ModelNet40
classes = [
    'airplane',
    'bathtub',
    'bed',
    'bench',
    'bookshelf',
    'bottle',
    'bowl',
    'car',
    'chair',
    'cone',
    'cup',
    'curtain',
    'desk',
    'door',
    'dresser',
    'flower_pot',
    'glass_box',
    'guitar',
    'keyboard',
    'lamp',
    'laptop',
    'mantel',
    'monitor',
    'night_stand',
    'person',
    'piano',
    'plant',
    'radio',
    'range_hood',
    'sink',
    'sofa',
    'stairs',
    'stool',
    'table',
    'tent',
    'toilet',
    'tv_stand',
    'vase',
    'wardrobe',
    'xbox'
]

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

def o3dvis_multi(pointclouds_list, auglevel_list=None):
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
        aug_level = auglevel_list[i]

        if max(auglevel_list) > 0:
            # a = aug_level/max(auglevel_list)
            a = aug_level/10
        else:
            a = 0
        # hue = 150 + a * 210
        hue = 200 + a * 160
        # hue = 340
        value = 0.9
        # sat = 0.5 + a * 0.3
        sat = 0.7

        hsv = np.array([hue, sat, value])
        rgb = hsv_to_rgb(hsv)
        
        # a = aug_level/max(auglevel_list)
        # # red = np.array([255 / 255., 0 / 255., 0 / 255.])
        # # cyan = np.array([0 / 255., 255 / 255., 255 / 255.])
        # rgb = np.array([(0 + 255*a)/255, 0/255, (255 - 128*a)/255])
        

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

def o3dvis(points_1, points_2, points_3=None, id=None, aug_level=None):
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

    chessboard_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0.1, 0.1, 0.1])


    color_1 = np.zeros_like(points_1)
    color_1[:] = blue_rgb
    P_1 = o3d.geometry.PointCloud()
    P_1.points = o3d.utility.Vector3dVector(points_1)
    P_1.colors = o3d.utility.Vector3dVector(color_1)

    red_hue = 0.0
    value = 1.0
    sat = 0.2 + (aug_level/10)*0.8
    hsv = np.array([red_hue, sat, value])
    rgb = hsv_to_rgb(hsv)
    

    points_2 = points_2 - [1, 0, 0]
    color_2 = np.zeros_like(points_2)
    # if aug_level is not None:
    #     color_2[:] = red_rgb*(10-aug_level)/10
    # else:
    color_2[:] = rgb
    # color_new[:] = green_rgb
    P_2 = o3d.geometry.PointCloud()
    P_2.points = o3d.utility.Vector3dVector(points_2)
    P_2.colors = o3d.utility.Vector3dVector(color_2)

    window_name = f"{id}"
    # o3d.visualization.draw_geometries([P_2], window_name=window_name, width=width, height=height,
    #                                   zoom=zoom, front=front, lookat=lookat, up=up)
    o3d.visualization.draw_geometries([P_1, P_2])

def load_h5_new(h5_name):
    f = h5py.File(h5_name, 'r')
    data_raw = f['raw'][:].astype('float32')
    # data_raw_wpointwolf = f['raw_pointwolf'][:].astype('float32')
    data_fake = f['pointcloud'][:].astype('float32')
    label = f['label'][:].astype('int64')
    accuaug_cnt = f['accuaug_cnt'][:].astype('int64')
    index = f['index'][:].astype('int64')
    f.close()
    return data_raw, data_fake, label, accuaug_cnt, index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1, help="epoch")
    parser.add_argument("--minibatch", type=int, default=0, help="minibatch")
    parser.add_argument("--root_path", type=str, default=None, help="log_path")
    parser.add_argument("--start", type=int, default=61, help="start")
    parser.add_argument("--end", type=int, default=None, help="end")
    parser.add_argument("--item", type=int, default=0, help="end")
    
    args = parser.parse_args()
    
    epoch = args.epoch
    minibatch = args.minibatch
    start = args.start
    if args.end is not None:
        end = args.end
    else:
        end = start+1
    # start = 151
    # end = 170
    
    if args.root_path is not None:
        root_path = args.root_path
    else:
        # root_path = f"/mnt/HDD8/max/TeachAugment_point/log/modelnet_dynamic_threshold/teachaugpoint_weight-pointnext-20240512-011203-CNqE7w2sah8CqWJLuRwVaM"
        # root_path = f"/mnt/HDD8/max/TeachAugment_point/log/modelnet_dynamic_threshold/teachaugpoint_weight-pointnext-20240512-185142-73HgpTj3gr8Pri6pWD8viZ"
        root_path = f"/mnt/HDD8/max/TeachAugment_point/log/modelnet_dynamic_threshold/teachaugpoint_weight-pointnext-20240522-143251-Dycdfmrveg72cNLNECFDri"
        # root_path = f"/mnt/HDD8/max/TeachAugment_point/log/modelnet_dynamic_threshold/teachaugpoint_weight-pointnext-20240627-230506-BqhcRCgTLspsewe253r38K"

    # path = f"{root_path}/fakedata/epoch61.h5"
    # data_raw, data_fake, labels, accuaug_cnts, indices = load_h5_new(h5_name=path)
    # for i in range(len(labels)):
    #     print(i, classes[int(labels[i])])

    pointclouds_list = []
    auglevel_list = []
    epochs = range(start, end)
    i = args.item
    print(i)
    for e in epochs:
        path = f"{root_path}/fakedata/epoch{e}.h5"
        if not os.path.exists(path):
            continue
        data_raw, data_fake, labels, accuaug_cnts, indices = load_h5_new(h5_name=path)
        
        raw = data_raw[i][:, :3]
        fake = data_fake[i][:, :3]
        label = labels[i]
        accuaug_cnt = accuaug_cnts[i]
        idex = indices[i]
        pointclouds = [
            {'name': 'raw', 'data': raw},
            {'name': 'fake', 'data': fake},
        ]
        print(f"{classes[int(label)]}", e, accuaug_cnt)
        pointclouds_list.append(fake)
        auglevel_list.append(accuaug_cnt)

    o3dvis_multi(pointclouds_list=pointclouds_list, auglevel_list=auglevel_list)

    # for i in range(data_raw.shape[0]):
    # i = 1
    # raw = data_raw[i][:, :3]
    # fake = data_fake[i][:, :3]
    # label = labels[i]
    # accuaug_cnt = accuaug_cnts[i]
    # pointclouds = [
    #     {'name': 'raw', 'data': raw},
    #     {'name': 'fake', 'data': fake},
    # ]
    # print(label, accuaug_cnt)
    # # o3dvis(points_1=raw, points_2=raw_wpointwolf, points_3=fake, id=f"{i}_{classes[int(label_)]}")
    # o3dvis(points_1=raw, points_2=fake, id=f"{i}_{classes[int(label)]}", aug_level=accuaug_cnt)

    #     front_size = [-0.078881283843093356, 0.98265290049739873, -0.16784224796908237]
    # lookat_size = [0.057118464194894254, -0.010695673712330742, 0.047245129152854129]
    # up_size = [0.018854223129214469, -0.16686616421723113, -0.98579927039414161]

        # camera_position = np.array([-0.5, -0.5, -0.5])
        # look_at = np.array([0.057118464194894254, -0.010695673712330742, 0.047245129152854129])
        # up_direction = np.array([0.018854223129214469, -0.16686616421723113, -0.98579927039414161])

        # for pc in pointclouds:
        #     name = pc['name']
        #     data = pc['data']
        #     data = data[:, [0, 2, 1]]
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')

        #     # Plotting the point cloud
        #     ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='.', s=1)

        #     # Set camera position and orientation
        #     ax.view_init(elev=30, azim=45)  # Adjust elevation and azimuth angles as needed
        #     ax.set_xlim([-1, 1])  # Adjust limits as needed
        #     ax.set_ylim([-1, 1])
        #     ax.set_zlim([-1, 1])
        #     ax.set_xlabel('X')
        #     ax.set_ylabel('Y')
        #     ax.set_zlabel('Z')

        #     # Save the rendered figure as an image (e.g., PNG format)
        #     plt.savefig(f'{newpath}/{i}_{classes[int(label_)]}_{name}.png', dpi=300)  # Adjust dpi as needed for image quality
        #     plt.close()


    # print('data_raw.shape:', data_raw.shape)
    # print('data_fake.shape:', data_fake.shape)
    # print('label.shape:', label.shape)








