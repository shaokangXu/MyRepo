'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-13 10:07:01
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-11-01 15:56:52
FilePath: /xushaokang/Single_AI_Registration/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
import torch
from model import Pose_Net
from torch.utils.data import DataLoader
from datasets import CustomDataset
from  SaveResultData import save_result
from Parameters import Read_Parameters
from util import SaveImage
import random
import os
device = torch.device('cuda')

def main(opt):
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    pose_net = Pose_Net()
    if opt.cuda:
        pose_net.cuda()
    #loda model
    if not os.path.exists(opt.Pose_Net):   
            print(opt.Pose_Net,"not exist")
            exit()
    else:
        pose_net.load_state_dict(torch.load(opt.Pose_Net))

    # Create a DataLoader
    with torch.no_grad():
        pose_net.eval()
        
        with open(opt.data_folder, 'r') as file:
        # 使用 len() 函数获取CT数量
            lines = file.readlines()
        #random.shuffle(lines) # 打乱数据集      
        for j,line in enumerate(lines):           
            print(line)
            train_loader = DataLoader(CustomDataset(line, opt.xray_per_ct, mode="test"), batch_size = opt.batch_size, shuffle=True, pin_memory=True)
            for i, (DRR_Single_Ap,X_ray_Single_Ap,DRR_Single_Lat,X_ray_Single_Lat,transform_parameters_delta,RotPointcord) in enumerate(train_loader):

                DRR_Single_Ap=DRR_Single_Ap.to(device)
                X_ray_Single_Ap=X_ray_Single_Ap.to(device)
                DRR_Single_Lat=DRR_Single_Lat.to(device)
                X_ray_Single_Lat=X_ray_Single_Lat.to(device)
                transform_parameters_delta=transform_parameters_delta.to(device)
                RotPointcord=RotPointcord.to(device)
                # Enter Network    
                start1=time.time()
                T,CatRotCenterCord=pose_net(DRR_Single_Ap,X_ray_Single_Ap,DRR_Single_Lat,X_ray_Single_Lat)
                #将回归值调整回原始参数
                #------
                #------
                T[:,-3:]*=100              
                print(CatRotCenterCord)
                print(RotPointcord)
                Time=round(time.time()-start1,3)  #计算配准时间
                #save_result(transform_parameters_delta,T,line,Time)  #保存相关输出数据信息(误差、配准时间)
                
                print("GT:",transform_parameters_delta)
                print("Predict:",T)
                print("Error",torch.round(abs(T-transform_parameters_delta),decimals=4))
                print("配准耗时:",Time,"秒")
                SaveImage(DRR_Single_Ap.squeeze(), 'Output/DRR_Single_Ap.png' )
                SaveImage(X_ray_Single_Ap.squeeze(), 'Output/X_ray_Single_Ap.png' )
                SaveImage(DRR_Single_Lat.squeeze(), 'Output/DRR_Single_Lat.png' )
                SaveImage(X_ray_Single_Lat.squeeze(), 'Output/X_ray_Single_Lat.png' )
                
            #exit()
            

if __name__ == '__main__':
    opt=Read_Parameters(mode='test') #读取测试参数
    main(opt)







