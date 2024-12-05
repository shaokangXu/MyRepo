'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-19 16:24:28
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-12-05 10:40:40
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
from util import Image_Process,SaveImage,draw_bboxes_with_names,get_T,point_in_which_bbox
from ParamTrans import Param_Trans
from segment import ExtractedByCoord
from ReadCenter import ReadCenter
from Parameters import common
from detect import detect
#from location import location,DrawLocationImage,map_names_to_bboxes
from point2plane import calculate_point_on_plane
opt = common()
sys.path.append('../Single_AI_Registration2.0/DRR_modules/')
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
    def __init__(self, ctd_txt_path, xray_per_ct,opt): 
        self.ctd_txt_path = ctd_txt_path.strip()
        self.mode=opt.mode
        self.CtName=os.path.basename(self.ctd_txt_path)[:-4]
        self.CT_file_path = '/home/ps/xushaokang/Datasets/CT-Lumbar/' +self.CtName+".nii.gz" #CT名字
        self.xray_per_ct = xray_per_ct

        self.Total_movImage, self.Total_movImageInfo = rw.ImageReader(self.CT_file_path, itk.Image[PixelType, Dimension])
    def __getitem__(self,index):
        flag=True
        while(flag):    
            thresholds = random.randint(-20,5) #Random projection threshold
            CT2DRR_rateAp= round(random.uniform(0.22,0.28),2)  #Randomly set the ratio of CT distance to light source and distance to imaging plane
            CT2DRR_rateLat= round(random.uniform(0.24,0.28),2) #0.26-0.32

            CtVolume_center=self.Total_movImageInfo["Volume_center"].copy()  #获取CT中心点       
            #---设置回归参数参照的旋转点（以投影位置为参考）
            ProReferPointAp=CtVolume_center.copy()
            ProReferPointLat=CtVolume_center.copy()
            ProReferPointAp[0]=ProReferPointAp[0]+random.randint(-20,20) ##(-40,40)
            ProReferPointAp[1]=ProReferPointAp[1]+projector_info['focal_lenght']*(CT2DRR_rateAp-0.2)
            ProReferPointAp[2]=ProReferPointAp[2]+random.randint(-20,20) ##(-30,30)

            ProReferPointLat[0]=ProReferPointLat[0]+projector_info['focal_lenght']*(CT2DRR_rateLat-0.2)
            ProReferPointLat[1]=ProReferPointLat[1]+random.randint(-20,20)  ##(-20,50)          
            ProReferPointLat[2]=ProReferPointLat[2]+random.randint(-20,20)  ##(-30,30)

            #---建立DRR投影 
            projectorAp = pm.projector_factory(projector_info, self.Total_movImage, self.Total_movImageInfo,PixelType,Dimension, 
                                                ScalarType,"AP",CT2DRR_rateAp,ProReferPointAp)
            projectorLat = pm.projector_factory(projector_info, self.Total_movImage, self.Total_movImageInfo,PixelType,Dimension, 
                                                ScalarType,"LAT",CT2DRR_rateLat,ProReferPointLat)            
            #---获取投影环境的相关信息
            ProInfoAp=projectorAp.getProInfo()
            ProInfoLat=projectorLat.getProInfo()
            
            #---读取椎体旋转中心点
            VertebralNameDictAp=ReadCenter(self.ctd_txt_path,ProReferPointAp[2])  #根据AP位投影获取CT中所有椎体质心
            VertebralNameDictLat=ReadCenter(self.ctd_txt_path,ProReferPointLat[2])  #根据LAT位投影获取CT中所有椎体质心
            
            #一、根据投影中心点选取配准椎体
            #---(1)找到键的交集
            common_Vertebral_keys = set(VertebralNameDictAp.keys()) & set(VertebralNameDictLat.keys())
            #---(2)根据交集的键来生成新字典
            common_Vertebral_dict = {key: VertebralNameDictAp[key] for key in common_Vertebral_keys}
            Vertebral_Name=random.choice(list(common_Vertebral_dict.keys())) #随机选择椎体名              
            Ver_Rot_center=np.array(VertebralNameDictAp[Vertebral_Name])   #获取对应椎体质心

            #二、合成图像数据
            #---(1)根据质心点随机生成初始参数、变换参数，并根据旋转点，计算目标参数
            transform_parameters_DRR,transform_parameters_delta,transform_parameters_X,RotCenterInit,RotCenterXray, RT_X, RT_init= Param_Trans(Ver_Rot_center)
            
            RotPointcord_DRR=calculate_point_on_plane(RotCenterInit,ProInfoAp["source"],ProInfoLat["source"],
                                                      ProInfoAp["DRRorigin"],ProInfoLat["DRRorigin"])
            
            RotPointcord_Xray=calculate_point_on_plane(RotCenterXray,ProInfoAp["source"],ProInfoLat["source"],
                                                      ProInfoAp["DRRorigin"],ProInfoLat["DRRorigin"])

            #---（2）整椎体DRR图像投影
            DRR_init_total_Ap = projectorAp.compute(transform_parameters_DRR, thresholds,Ver_Rot_center)  # AP  Proj
            DRR_init_total_Ap = Image_Process(DRR_init_total_Ap, opt.image_size).to(device)   
            DRR_init_total_Ap = 0.5*(DRR_init_total_Ap+1)## 归一化

            DRR_init_total_Lat = projectorLat.compute(transform_parameters_DRR, thresholds,Ver_Rot_center)  # AP  Proj  
            DRR_init_total_Lat = Image_Process(DRR_init_total_Lat, opt.image_size).to(device)   
            DRR_init_total_Lat = 0.5*(DRR_init_total_Lat+1)## 归一化  
            
            # 合成X-ray图像    
            X_ray_DRR_Ap = projectorAp.compute(transform_parameters_X, thresholds,Ver_Rot_center)  # X-ray AP  Proj
            X_ray_DRR_Ap  = Image_Process(X_ray_DRR_Ap , opt.image_size).to(device)      
            X_ray_Ap = Style_trans(X_ray_DRR_Ap , "AP") ## 风格迁移

            X_ray_DRR_Lat = projectorLat.compute(transform_parameters_X, thresholds,Ver_Rot_center)  # X-ray AP  Proj
            X_ray_DRR_Lat  = Image_Process(X_ray_DRR_Lat , opt.image_size).to(device)      
            X_ray_Lat = Style_trans(X_ray_DRR_Lat , "LAT") ## 风格迁移 
            #----------------------------------  

            #-----image Detect------
            DRR_AP_location = detect(1-DRR_init_total_Ap,"AP","DRR_Ap_only_detect.png",DrawImage=False)
            X_ray_AP_location = detect(1-(X_ray_Ap+1)*0.5,"AP","X-ray_Ap_only_detect.png",DrawImage=False)
            DRR_LAT_location = detect(1-DRR_init_total_Lat,"LAT","DRR_Lat_only_detect.png",DrawImage=False)
            X_ray_LAT_location = detect(1-(X_ray_Lat+1)*0.5,"LAT","X-ray_Lat_only_detect.png",DrawImage=False)

            DRRVerchooseLocationAp = point_in_which_bbox(RotPointcord_DRR["AP"],DRR_AP_location)
            DRRVerchooseLocationLat = point_in_which_bbox(RotPointcord_DRR["LAT"],DRR_LAT_location)
            X_rayVerchooseLocationAp = point_in_which_bbox(RotPointcord_Xray["AP"],X_ray_AP_location)      
            X_rayVerchooseLocationLat = point_in_which_bbox(RotPointcord_Xray["LAT"],X_ray_LAT_location)
            
            if(DRRVerchooseLocationAp==[]or X_rayVerchooseLocationAp==[] or DRRVerchooseLocationLat==[] or X_rayVerchooseLocationLat==[] ):
                #print(self.CT_file_path,"is No")
                flag = True
                projectorAp.delete()
                projectorLat.delete()
                continue
            else:
                flag = False

            #-----Divide-----
            #根据坐标提取DRR和Xray单椎体部分（返回值：单椎体图像，区域顶部位置）
            DRR_Single_Ap  = ExtractedByCoord(1-DRR_init_total_Ap,DRRVerchooseLocationAp,False,"AP")
            DRR_Single_Lat = ExtractedByCoord(1-DRR_init_total_Lat,DRRVerchooseLocationLat,False,"LAT")
            k=random.choice([0,0,1,1,1])
            if k: 
                occlusion = True           
            else: 
                occlusion = False
            X_ray_Single_Ap = ExtractedByCoord(1-X_ray_Ap,X_rayVerchooseLocationAp,occlusion,View="AP")            
            X_ray_Single_Lat = ExtractedByCoord(1-X_ray_Lat,X_rayVerchooseLocationLat,occlusion,View="LAT")
        
        transform_parameters_delta = torch.tensor(transform_parameters_delta).view(6).float()
        RotPointcord=torch.tensor(RotPointcord_DRR["AP"]+RotPointcord_Xray["AP"]+RotPointcord_DRR["LAT"]+RotPointcord_Xray["LAT"]).view(8).float() / 512
        projectorAp.delete()
        projectorLat.delete()
        if self.mode=='train':
            return  DRR_Single_Ap , X_ray_Single_Ap , DRR_Single_Lat , X_ray_Single_Lat , transform_parameters_delta,RotPointcord
        else:
            return  DRR_Single_Ap , X_ray_Single_Ap , DRR_Single_Lat , X_ray_Single_Lat , transform_parameters_delta, RotPointcord, RotCenterInit, RT_X, RT_init, Vertebral_Name


    def __len__(self):
        return self.xray_per_ct

