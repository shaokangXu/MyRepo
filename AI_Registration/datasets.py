'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-19 16:24:28
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-08 15:21:10
FilePath: /xushaokang/AI_Registration1/datasets.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from torch.utils.data import Dataset
import itk
import argparse
import torch
import random
import sys
import numpy as np
from Style_trans import Style_trans
import time
from utils import Image_Process,get_T
sys.path.append('../AI_Registration4/DRR_modules/')
import ProjectorsModule as pm
import ReadWriteImageModule as rw
import RigidMotionModule as rm

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default= True, help='use GPU computation')
parser.add_argument('--image_size', type=int, default=512)
opt = parser.parse_args()
device = torch.device('cuda')
#Projector params
projector_info = {
    'Name': 'SiddonGpu',
    'threadsPerBlock_x': 16,
    'threadsPerBlock_y': 16,
    'threadsPerBlock_z': 1,
    'DRRsize': 1024,
    'DRRspacing': 0.209,
    'DRR_pp': 0.0,
    'focal_lenght': 1011.0
}



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ct_txt_path, xray_per_ct):
        with open(ct_txt_path, 'r') as f:
            self.ct_paths = list(map(lambda line: line.strip().split(' '), f))
            self.xray_per_ct = xray_per_ct

    def __getitem__(self, index):

        ct_path = self.ct_paths[int(index/self.xray_per_ct)]
        PixelType = itk.F
        Dimension = 3
        ScalarType = itk.D
        thresholds = random.randint(-30,10)  # 随机设置投影阈值
        # 建立正侧位DRR投影环境
        movImage, movImageInfo = rw.ImageReader(ct_path[0], itk.Image[PixelType, Dimension]) #读取CT数据
        projector_AP = pm.projector_factory(projector_info, movImage, movImageInfo, PixelType, Dimension, ScalarType,"AP")  
        projector_LAT = pm.projector_factory(projector_info, movImage, movImageInfo, PixelType, Dimension, ScalarType,"LAT")
        # 初始 DRR 参数
        rx_init = round(random.uniform(-0.187, 0.187),4)
        ry_init = round(random.uniform(-0.187, 0.187),4)
        rz_init = round(random.uniform(-0.187, 0.187),4)
        tx_init = round(random.uniform(-3, 3),2)
        ty_init = round(random.uniform(-3, 3),2)
        tz_init = round(random.uniform(-3, 3),2)
        transform_parameters_init = [rx_init,ry_init, rz_init, tx_init, ty_init, tz_init]
        
        # 变换 DRR 参数
        delta_rx =round(random.uniform(-0.174, 0.174),4)
        delta_ry = round(random.uniform(-0.174, 0.174),4)
        delta_rz = round(random.uniform(-0.174, 0.174),4)
        delta_tx = round(random.uniform(-5, 5),2)
        delta_ty = round(random.uniform(-5, 5),2)
        delta_tz = round(random.uniform(-5, 5),2)
       
        transform_parameters_delta = [delta_rx, delta_ry, delta_rz,delta_tx,delta_ty, delta_tz]
        
        # 计算合成X-ray图像的变换参数
        # 1、计算6个参数对应的RT矩阵
        RT_init = rm.get_rigid_motion_mat_from_euler(rx_init, 'x', ry_init, 'y', rz_init, 'z', tx_init, ty_init,
                                                        tz_init, movImageInfo['Volume_center'])
        RT_delta = rm.get_rigid_motion_mat_from_euler(delta_rx, 'x', delta_ry, 'y', delta_rz, 'z', delta_tx,
                                                        delta_ty, delta_tz, movImageInfo['Volume_center'])

        RT_X = np.dot(RT_delta, RT_init)
        transform_parameters_X_ray = get_T(RT_X, movImageInfo['Volume_center'])  #根据RT矩阵计算对应6个参数


        # 投影初始位置的DRR图像
        DRR_AP_init = projector_AP.compute(transform_parameters_init, thresholds)  # AP  Proj
        DRR_LAT_init = projector_LAT.compute(transform_parameters_init, thresholds)  # LAT  Proj
        
        DRR_AP_init = Image_Process(DRR_AP_init, opt.image_size).squeeze(0)
        DRR_LAT_init = Image_Process(DRR_LAT_init, opt.image_size).squeeze(0)

        # 投影X-ray图像
        X_ray_AP = projector_AP.compute(transform_parameters_X_ray, thresholds)  # X-ray AP  Proj
        X_ray_LAT = projector_LAT.compute(transform_parameters_X_ray, thresholds)  # X-ray LAT  Proj
        
        X_ray_AP = Image_Process(X_ray_AP, opt.image_size).to(device)
        X_ray_LAT = Image_Process(X_ray_LAT, opt.image_size).to(device)

        # 风格迁移       
        X_ray_AP = Style_trans(X_ray_AP, "AP").to('cpu').squeeze(0)
        X_ray_LAT = Style_trans(X_ray_LAT, "LAT").to('cpu').squeeze(0)

        #释放投影环境占用的内存
        projector_AP.delete()
        projector_LAT.delete()
        
        transform_parameters_delta = torch.tensor(transform_parameters_delta).view(6)
         
        
        return DRR_AP_init, X_ray_AP, DRR_LAT_init,X_ray_LAT,transform_parameters_delta

    def __len__(self):
        return len(self.ct_paths * self.xray_per_ct)



""" import os

# 指定文件夹的路径
folder_path = '../Datasets/CT-test/'

# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)

# 创建一个新的文本文件用于保存文件路径
txt_file_path = '../Datasets/ct_test.txt'

# 打开文本文件以写入模式
with open(txt_file_path, 'w') as txt_file:
    # 遍历文件夹中的文件名，并将完整路径写入文本文件
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        txt_file.write(file_path + '\n')

print(f'文件路径已保存到 {txt_file_path}') """