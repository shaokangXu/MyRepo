'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-29 15:23:33
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-08 15:21:13
FilePath: /xushaokang/MaestroAlgoImgRegisDL/AI_Registration/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
from torchvision.utils import save_image
import argparse
import torch
from models import Pose_Net
from torch.utils.data import DataLoader
from datasets import CustomDataset
from SaveResultData import save_result


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default= True, help='use GPU computation')
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--Pose_Net', type=str, default='train_model/Pose_Net_100.pth', help='AP_CNN checkpoint file')
opt = parser.parse_args()

device = torch.device('cuda')
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
pose_net = Pose_Net()
if opt.cuda:
    pose_net.cuda()

#loda model
pose_net.load_state_dict(torch.load(opt.Pose_Net))
pose_net.eval()

# Create a DataLoader
data_folder = '../Datasets/ct_test.txt'
batch_size = 1
xray_per_ct = 1
train_loader = DataLoader(CustomDataset(data_folder, xray_per_ct), batch_size = batch_size, shuffle=True, pin_memory=True)

#pose_net.eval()
Error=[0,0,0,0,0,0]
for i, (DRR_AP_init, X_ray_AP, DRR_LAT_init,X_ray_LAT,transform_parameters_delta) in enumerate(train_loader):
    # Label numpy to tensor
    start1=time.time()
    DRR_AP_init=DRR_AP_init.to(device)
    X_ray_AP=X_ray_AP.to(device)
    DRR_LAT_init=DRR_LAT_init.to(device)
    X_ray_LAT=X_ray_LAT.to(device)
    transform_parameters_delta=transform_parameters_delta.to(device)  

    # Enter Network         
    R,T= pose_net(DRR_AP_init,DRR_LAT_init,X_ray_AP,X_ray_LAT)                   
    delta_Pose=torch.cat((R,T),dim=1)
    delta_Pose[:,-3:] *=100
    print("Pre:",delta_Pose)
    print("GT:",transform_parameters_delta)
    Time=time.time()-start1
    #保存相关输出数据信息(误差、配准时间)
    save_result(transform_parameters_delta,delta_Pose,i,Time)






