'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-13 10:07:01
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-06-12 14:21:13
FilePath: /xushaokang/Single_AI_Registration/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
from torchvision.utils import save_image
import argparse
import torch
from model import Pose_Net
import torch.nn as nn
from Style_trans import Style_trans
import os
from util import Image_Process,get_T
from segment import extracted
import sys
import random
import itk
import numpy as np
sys.path.append('../Single_AI_Registration2.0_BN/DRR_modules/')
import ProjectorsModule as pm
import ReadWriteImageModule as rw
import RigidMotionModule as rm


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default= True, help='use GPU computation')
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--Pose_Net', type=str, default='checkpoint_model/Pose_Net_Single_AP241.pth', help='checkpoint file')
opt = parser.parse_args()

start1=time.time()
device = torch.device('cuda')
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
pose_net = Pose_Net()
if opt.cuda:
    pose_net.cuda()

#loda model
pose_net.load_state_dict(torch.load(opt.Pose_Net))
pose_net.eval()

print(">>>完成配准模型导入---耗时:",time.time()-start1,"s")

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

PixelType = itk.F
Dimension = 3
ScalarType = itk.D
# Create a DataLoader
data_folder = '../Single_Dataset/ct_Single_test_others.txt'


with torch.no_grad():
    with open(data_folder, 'r') as file:
    # 使用 len() 函数获取CT数量
        lines = file.readlines()      
        for j,line in enumerate(lines):
            start2=time.time()           
            print(line)
            SingleCT_path = line.strip()#获取单椎体路径
            view="AP"
            file_name = os.path.basename(os.path.dirname(SingleCT_path)) #CT名字
            #num = file_name.split('.nii.gz')[0] #获取单椎体编号，str类型
            TotalCT_path="../Single_Dataset/CT/"+file_name+".nii.gz" #获取整椎体路径
            Single_movImage, Single_movImageInfo = rw.ImageReader(SingleCT_path, itk.Image[PixelType, Dimension])
            Total_movImage, Total_movImageInfo = rw.ImageReader(TotalCT_path, itk.Image[PixelType, Dimension])
            print(">>>完成单椎体、整椎体CT读取---耗时:",time.time()-start2,"s")


            start3=time.time()
            thresholds = random.randint(-30,0) #随机设置投影阈值
            CT2DRR_rate= round(random.uniform(1/9,1/5),2)  #随机设置两个视图中CT距离光源与距离成像平面的比例
            ## -------生成DRR图像-----------
            #读取 单 椎体CT信息并建立DRR投影      
            projector_single = pm.projector_factory(projector_info, Single_movImage, Single_movImageInfo, PixelType, Dimension, ScalarType,view,CT2DRR_rate)
            #读取 整 椎体CT信息并建立DRR投影 
            projector_total = pm.projector_factory(projector_info, Total_movImage, Total_movImageInfo, PixelType, Dimension, ScalarType,view,CT2DRR_rate)
            print(">>>完成两个投影坐标系的建立---耗时:",time.time()-start3,"s")
            #选取旋转中心点
            """ Volume_center=ReadCenter(WholeCT_Name,int(num))  #计算单椎体质心
            Volume_center=np.array(Volume_center) """



            start4=time.time()
            #设置两个视图的旋转点
            Rot_center=Total_movImageInfo["Volume_center"].copy()
            if view=="AP":
                Rot_center[1]=Rot_center[1]+projector_info['focal_lenght']*(CT2DRR_rate-0.1)
            else:
                Rot_center[0]=Rot_center[0]+projector_info['focal_lenght']*(CT2DRR_rate-0.1)

            #随机生成初始参数、变换参数，并根据旋转点，计算目标参数
            rx_DRR = round(random.uniform(-0.05, 0.05),3)
            ry_DRR = round(random.uniform(-0.05, 0.05),3)
            rz_DRR = round(random.uniform(-0.05, 0.05),3)
            tx_DRR = round(random.uniform(-8, 8),1)
            ty_DRR = round(random.uniform(-8, 8),1)
            tz_DRR = round(random.uniform(-8, 8),1)  
            """ rx_DRR = 0
            ry_DRR = 0
            rz_DRR = 0
            tx_DRR = 0
            ty_DRR = 0
            tz_DRR = 0 """
            transform_parameters_DRR = [rx_DRR, ry_DRR, rz_DRR,tx_DRR,ty_DRR, tz_DRR]

            # 随机产生初定位与X-ray之间投影变换参数差值  
            delta_rx =round(random.uniform(-0.1, 0.1),3)
            delta_ry = round(random.uniform(-0.1, 0.1),3)
            delta_rz = round(random.uniform(-0.1, 0.1),3)
            delta_tx = round(random.uniform(-5, 5),1)
            delta_ty = round(random.uniform(-5, 5),1)
            delta_tz = round(random.uniform(-5, 5),1)
            """ delta_rx = 0
            delta_ry = 0
            delta_rz = 0
            delta_tx = 0
            delta_ty = 5
            delta_tz = 0 """
            transform_parameters_delta = [delta_rx, delta_ry, delta_rz,delta_tx,delta_ty, delta_tz]

            RT_init = rm.get_rigid_motion_mat_from_euler(transform_parameters_DRR[0], 'x', 
                                                        transform_parameters_DRR[1], 'y', 
                                                        transform_parameters_DRR[2], 'z', 
                                                        transform_parameters_DRR[3], 
                                                        transform_parameters_DRR[4],
                                                        transform_parameters_DRR[5], 
                                                        Rot_center)
            RT_delta = rm.get_rigid_motion_mat_from_euler(transform_parameters_delta[0], 'x', 
                                                        transform_parameters_delta[1], 'y', 
                                                        transform_parameters_delta[2], 'z', 
                                                        transform_parameters_delta[3],
                                                        transform_parameters_delta[4], 
                                                        transform_parameters_delta[5], 
                                                        Rot_center)

            RT_X = np.dot(RT_delta, RT_init)  #旋转矩阵变换
            transform_parameters_X = get_T(RT_X, Rot_center)#根据旋转矩阵以及旋转点计算6个变换参数
            print(">>>完成投影参数初始化及合成参数计算---耗时:",time.time()-start4,"s")



            start5=time.time()
            #---------------image-------------------
            # 单椎体DRR图像投影
            DRR_init_single = projector_single.compute(transform_parameters_DRR, thresholds,Rot_center)  # AP  Proj 
            DRR_init_single = Image_Process(DRR_init_single, opt.image_size).to(device)
            DRR_init_single = 0.5*(DRR_init_single+1)## 归一化
            
            DRR_X_single = projector_single.compute(transform_parameters_X, thresholds,Rot_center)  # AP  Proj 
            DRR_X_single = Image_Process(DRR_X_single, opt.image_size).to(device)
            DRR_X_single = 0.5*(DRR_X_single+1)## 归一化

            # 整椎体DRR图像投影
            DRR_init_total = projector_total.compute(transform_parameters_DRR, thresholds,Rot_center)  # AP  Proj 
            DRR_init_total = Image_Process(DRR_init_total, opt.image_size).to(device)   
            DRR_init_total = 0.5*(DRR_init_total+1)## 归一化  
            # 合成X-ray图像    
            X_ray = projector_total.compute(transform_parameters_X, thresholds,Rot_center)  # X-ray AP  Proj
            X_ray1 = Image_Process(X_ray, opt.image_size).to(device)      
            X_ray_T = Style_trans(X_ray1 , view)## 风格迁移

            #--X-ray detect---
            """ AP_location = detect(X_ray_AP,"AP")
            LAT_location = detect(X_ray_LAT,"LAT")

            segmentation(X_ray_AP,AP_location,2,"AP")
            segmentation(X_ray_LAT,LAT_location,2,"LAT") """
            
            #根据单椎体的位置，划分出对应DRR图像与X-ray图像中对应椎体的区域
            X_ray=extracted(DRR_X_single,1-X_ray_T).to(device)
            DRR_init=extracted(DRR_init_single,1-DRR_init_total).to(device)
     
            transform_parameters_delta = torch.tensor(transform_parameters_delta).view(6)
            transform_parameters_delta=transform_parameters_delta.to(device) 
            print(">>>完成DRR图像及X-ray图像生成---耗时:",time.time()-start5,"s")


            
            start6=time.time()             
            # Enter Network          
            T= pose_net(DRR_init,X_ray)   #预测
            print(">>>完成配准模型推理---耗时:",time.time()-start6,"s")
            delta_Pose=T
            #delta_Pose[:,:3]/=10 
            delta_Pose[:,-3:]*=100 
            print("Pre:",delta_Pose)
            print("GT:",transform_parameters_delta)
            Time=round(time.time()-start1,3)
            #save_result(transform_parameters_delta,delta_Pose,i,Time)

            ####根据预测参数计算配准图像
            delta_Pose=delta_Pose.to('cpu')
            delta_Pose=delta_Pose[0].tolist()
            
            pre_RT_delta = rm.get_rigid_motion_mat_from_euler(delta_Pose[0], 'x', delta_Pose[1], 'y', delta_Pose[2], 'z', delta_Pose[3],
                                                            delta_Pose[4], delta_Pose[5], Rot_center)
            RT_reg = np.dot(pre_RT_delta, RT_init)
            transform_parameters_X = get_T(RT_reg, Rot_center)#根据旋转矩阵以及旋转点计算6个变换参数
            
            # 配准好的单椎体DRR图像投影
            DRR = projector_single.compute(transform_parameters_X, thresholds,Rot_center)  # AP  Proj
            DRR = Image_Process(DRR, opt.image_size).to(device)
            ## 归一化
            X_ray1 = 0.5*(X_ray1+1)
            DRR = 0.5*(DRR+1)
            #save_image(DRR_init_total, 'Output/DRR_init_total'+view+'.png' )
            save_image(DRR_init, 'Output/DRR_init.png' )
            #save_image(X_ray1, 'Output/X_ray1'+view+'.png' )
            save_image(X_ray, 'Output/X_ray.png' )
            save_image(1-DRR, 'Output/DRR.png' )

            projector_single.delete()
            projector_total.delete()

            with open("chessboard.py","r") as F:
                    code=F.read()
            exec(code)
            exit()

    
    






