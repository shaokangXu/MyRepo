'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-19 16:24:28
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-06-12 15:01:07
FilePath: /xushaokang/AI_Registration1/datasets.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torchvision.utils import save_image
from torch.utils.data import Dataset
import itk
import torch
import random
import sys
import numpy as np
from Style_trans import Style_trans
import os
from util import Image_Process
from ParamTrans import Param_Trans
from segment import extracted_retangle
from ReadCenter import ReadCenter
from Parameters import common
opt = common()
sys.path.append('../Single_AI_Registration2.0_BN/DRR_modules/')
import ProjectorsModule as pm
import ReadWriteImageModule as rw

device = torch.device('cuda')

#DRR投影的相关参数
projector_info = {
    'Name': 'SiddonGpu',
    'threadsPerBlock_x': 16,
    'threadsPerBlock_y': 16,
    'threadsPerBlock_z': 1,
    'DRRsize': 1024,
    'DRRspacing': 0.209,
    'DRR_pp': 0.0,
    'focal_lenght': 1060.0
}

PixelType = itk.F
Dimension = 3
ScalarType = itk.D


class CustomDataset(Dataset):
    def __init__(self, ct_txt_path, xray_per_ct, view):
        SingleCT_path = ct_txt_path.strip()#获取单椎体路径
        self.xray_per_ct = xray_per_ct
        self.view=view
        file_name = os.path.basename(os.path.dirname(SingleCT_path)) #CT名字
        #num = file_name.split('.nii.gz')[0] #获取单椎体编号，str类型
        TotalCT_path="../Single_Dataset/CT/"+file_name+".nii.gz" #获取整椎体路径
        self.Single_movImage, self.Single_movImageInfo = rw.ImageReader(SingleCT_path, itk.Image[PixelType, Dimension])
        self.Total_movImage, self.Total_movImageInfo = rw.ImageReader(TotalCT_path, itk.Image[PixelType, Dimension])

    def __getitem__(self,index):    
        thresholds = random.randint(-30,0) #随机设置投影阈值
        CT2DRR_rate= round(random.uniform(1/9,1/5),2)  #随机设置两个视图中CT距离光源与距离成像平面的比例

        ## -------生成DRR图像-----------
        #读取 单 椎体CT信息并建立DRR投影      
        projector_single = pm.projector_factory(projector_info, self.Single_movImage, self.Single_movImageInfo, PixelType, Dimension, ScalarType,self.view,CT2DRR_rate)
        #读取 整 椎体CT信息并建立DRR投影 
        projector_total = pm.projector_factory(projector_info, self.Total_movImage, self.Total_movImageInfo, PixelType, Dimension, ScalarType,self.view,CT2DRR_rate)
        
        #选取旋转中心点
        """ Volume_center=ReadCenter(WholeCT_Name,int(num))  #计算单椎体质心
        Volume_center=np.array(Volume_center) """
        #设置两个视图的旋转点
        Rot_center=self.Total_movImageInfo["Volume_center"].copy()
        if self.view=="AP":
            Rot_center[1]=Rot_center[1]+projector_info['focal_lenght']*(CT2DRR_rate-0.1)
        else:
            Rot_center[0]=Rot_center[0]+projector_info['focal_lenght']*(CT2DRR_rate-0.1)

        #随机生成初始参数、变换参数，并根据旋转点，计算目标参数
        transform_parameters_DRR,transform_parameters_delta,transform_parameters_X = Param_Trans(Rot_center)
       
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
        X_ray = Image_Process(X_ray, opt.image_size).to(device)      
        X_ray = Style_trans(X_ray , self.view)## 风格迁移 
        #----------------------------------  

        #-----X-ray detect------
        """ AP_location = detect(X_ray_AP,"AP")
        LAT_location = detect(X_ray_LAT,"LAT")

        segmentation(X_ray_AP,AP_location,2,"AP")
        segmentation(X_ray_LAT,LAT_location,2,"LAT") """
        
        #根据单椎体的位置，划分出对应DRR图像与X-ray图像中对应椎体的区域
        X_ray=extracted_retangle(DRR_X_single,1-X_ray).squeeze(0)
        DRR_init=extracted_retangle(DRR_init_single,1-DRR_init_total).squeeze(0)

        transform_parameters_delta = torch.tensor(transform_parameters_delta).view(6)


        """ save_image(DRR_init_single, 'Output/DRR_init_single_'+self.view+'.png' )
        save_image(DRR_init, 'Output/DRR_init_'+self.view+'.png' )
        save_image(X_ray, 'Output/X_ray_'+self.view+'.png' )
        exit() """

        projector_single.delete()
        projector_total.delete()
        return DRR_init, X_ray,transform_parameters_delta

        
    def __len__(self):
        return self.xray_per_ct

