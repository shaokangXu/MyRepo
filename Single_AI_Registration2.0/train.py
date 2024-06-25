'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-12 11:37:54
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-29 15:18:32
FilePath: /xushaokang/AI_Registration1/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import Pose_Net
from torch.utils.data import DataLoader
from datasets import CustomDataset
from util import Save_train_info
import os
from Parameters import Read_Parameters
import random
device = torch.device('cuda')

def main(opt):
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    pose_net = Pose_Net(opt.num_classes)
    if opt.cuda:
        pose_net.cuda()

    #判断模型文件是否存在
    if opt.load_dict:
        print("===============")
        # Load state dicts
        if not os.path.exists(opt.Pose_Net):   
            print(opt.Pose_Net,"not exist")
            exit()
        pose_net.load_state_dict(torch.load(opt.Pose_Net))

    criterion = nn.MSELoss() #损失函数
    optimizer = optim.Adam(pose_net.parameters(),lr=opt.lr, betas=(0.5, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

    #掩盖对应视图不易回归的参数
    mask = torch.ones(1, 6).to(device)
    if opt.view=='AP':      
       mask[:, 4] = 0
    else:
       mask[:, 3] = 0

    # Create a DataLoader   
    if not os.path.exists(opt.data_folder):   #判断数据集文件是否存在
        print(opt.data_folder,"not exist")
        exit()

    
    for epoch in range(opt.start_epoch ,opt.epochs):
        
        total_loss = 0.0
        num_batches = 0

        with open(opt.data_folder, 'r') as file:           
            # 使用 len() 函数获取CT数量
            lines = file.readlines()
            CT_num = len(lines)      
        for j,line in enumerate(lines):            
            # 随机选择一行
            """ line = random.sample(lines, 1)[0]    
            lines.remove(str(random_line)) """
            train_loader = DataLoader(CustomDataset(line, opt.xray_per_ct,opt.view), batch_size = opt.batch_size, shuffle=True, pin_memory=True)
            for i, (DRR_AP_init, X_ray_AP,transform_parameters_delta) in enumerate(train_loader):
                pose_net.train()####
                DRR_AP_init=DRR_AP_init.to(device)
                X_ray_AP=X_ray_AP.to(device)

                transform_parameters_delta=transform_parameters_delta.to(device)
                # Enter Network    
                optimizer.zero_grad()         
                delta_T=pose_net(DRR_AP_init,X_ray_AP)
                #将groundtruth参数调整至[-1,1]
                transform_parameters_delta[:,:3]*=10 
                transform_parameters_delta[:,-3:]/=5               
                Loss= criterion(delta_T*mask,transform_parameters_delta*mask)
                Loss.backward()
                optimizer.step()                     
                print(opt.view+"  Epoch [{}/{}] Now[{}/{}] , Loss: {:0.6f}".format(epoch,opt.epochs,int(j*(opt.xray_per_ct/opt.batch_size)+i),int((CT_num*opt.xray_per_ct)/opt.batch_size),  Loss.item()))            
                # 累加损失和批次数
                total_loss += Loss.item()
                num_batches += 1
                
        
        average_loss = total_loss / num_batches  # 计算每个epoch的平均损失
        current_learning_rate = optimizer.param_groups[0]['lr']  # 获取当前学习率的值
        Save_train_info(epoch,opt.epochs,average_loss,current_learning_rate,opt.view) #保存训练相关信息（loss,lr...）

        #保存最近5次的模型  
        model_path='checkpoint_model/Pose_Net_Single_'+opt.view
        torch.save(pose_net.state_dict(), model_path+str(epoch)+'.pth') 
        if epoch>5 and  os.path.exists(model_path+str(epoch-5)+'.pth'):
            os.remove(model_path+str(epoch-5)+'.pth')
        scheduler.step()

    
if __name__ == '__main__':
    opt=Read_Parameters(mode='train') #读取训练参数
    
    main(opt)