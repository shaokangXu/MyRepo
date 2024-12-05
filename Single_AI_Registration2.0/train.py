'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-12 11:37:54
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-11-26 16:03:09
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
from util import Save_train_info,LinearDecayScheduler
import os
from Parameters import Read_Parameters
import inspect
device = torch.device('cuda')

def main(opt):
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    pose_net = Pose_Net(opt.modelName)
    if opt.cuda:
        pose_net.cuda()

    #判断模型文件是否存在
    if opt.load_weight:
        print("===============")
        # Load state dicts
        if not os.path.exists(opt.Pose_Net):   
            line=inspect.currentframe().f_lineno  #get current line number
            exit(__file__+" line:"+str(line)+" "+opt.Pose_Net+" not exist !!")
        #模型有改动时，可以加载未改动的部分继续训练
        """ pretrained_model = torch.load(opt.Pose_Net)
        pose_net_state_dict = pose_net.state_dict()
        # 定义你要加载的部分
        partial_state_dict = {k: v for k, v in pretrained_model.items() if k in pose_net_state_dict}
        print(partial_state_dict.keys())
        # 加载这些部分的权重
        pose_net_state_dict.update(partial_state_dict)
        pose_net.load_state_dict(pose_net_state_dict) """
        pose_net.load_state_dict(torch.load(opt.Pose_Net))

    criterion = nn.MSELoss().to(device) #损失函数
    optimizer = optim.Adam(pose_net.parameters(),lr=opt.lr, betas=(0.5, 0.999))
    scheduler = LinearDecayScheduler(optimizer, start_epoch=opt.decay_epochs, total_epochs=opt.epochs, initial_lr=opt.lr)

    # Create a DataLoader   
    if not os.path.exists(opt.data_folder):   #判断数据集文件是否存在
        line=inspect.currentframe().f_lineno  #get current line number
        exit(__file__+" line:"+str(line)+" "+opt.data_folder+" not exist !!")

    
    for epoch in range(opt.start_epoch ,opt.epochs):
        pose_net.train()
        total_loss = 0.0
        num_batches = 0
        with open(opt.data_folder, 'r') as file:
            CT_num = len(file.readlines())
        with open(opt.data_folder, 'r') as file:
        # 使用 len() 函数获取CT数量
            lines = file.readlines()      
        #random.shuffle(lines) # 打乱数据集
        for j,line in enumerate(lines):    
            train_loader = DataLoader(CustomDataset(line, opt.xray_per_ct,opt), batch_size = opt.batch_size, shuffle=True, pin_memory=True)
            for i, (DRR_Single_Ap,X_ray_Single_Ap,DRR_Single_Lat,X_ray_Single_Lat,transform_parameters_delta,RotPointcord) in enumerate(train_loader):

                DRR_Single_Ap=DRR_Single_Ap.to(device)
                X_ray_Single_Ap=X_ray_Single_Ap.to(device)
                DRR_Single_Lat=DRR_Single_Lat.to(device)
                X_ray_Single_Lat=X_ray_Single_Lat.to(device)
                transform_parameters_delta=transform_parameters_delta.to(device)
                RotPointcord=RotPointcord.to(device)
                # Enter Network    
                optimizer.zero_grad()         
                delta_T,CatRotCenterCord=pose_net(DRR_Single_Ap, X_ray_Single_Ap, DRR_Single_Lat, X_ray_Single_Lat)
                
                # Loss
                transform_parameters_delta[:,-3:]/=100
                Loss_param= criterion(delta_T,transform_parameters_delta)*10                         
                Loss_center= criterion(CatRotCenterCord,RotPointcord)
                #Loss_Recons= criterion(ReconsXrayAp,X_ray_Single_Ap)+criterion(ReconsXrayLat,X_ray_Single_Lat)
                loss_total=Loss_param+Loss_center
                loss_total.backward()
                
                optimizer.step()
                print(opt.modelName+" Epoch [{}/{}] Now[{}/{}], Loss_param: {:0.6f}, Loss_center: {:0.6f}".format(epoch,opt.epochs,int(j*(opt.xray_per_ct/opt.batch_size)+i),int((CT_num*opt.xray_per_ct)/opt.batch_size), Loss_param.item(), Loss_center.item()))                      
                     
                # 累加损失和批次数
                total_loss += loss_total.item()
                num_batches += 1
            if j % 100 == 0:
                model_path='checkpoint_model/Pose_Net_Single_occ_'+opt.modelName
                torch.save(pose_net.state_dict(), model_path+"_"+str(epoch)+'.pth')     
        average_loss = total_loss / num_batches  # 计算每个epoch的平均损失
        current_learning_rate = optimizer.param_groups[0]['lr']  # 获取当前学习率的值
        Save_train_info(epoch,opt.epochs,average_loss,current_learning_rate,opt.LossDataName+opt.modelName) #保存训练相关信息（loss,lr...）

        #保存最近5次的模型  
        model_path='checkpoint_model/Pose_Net_Single_occ_'+opt.modelName
        torch.save(pose_net.state_dict(), model_path+"_"+str(epoch)+'.pth') 
        if epoch>5 and  os.path.exists(model_path+"_"+str(epoch-5)+'.pth'):
            os.remove(model_path+"_"+str(epoch-5)+'.pth')     
        scheduler.step(epoch)

if __name__ == '__main__':
    opt=Read_Parameters(mode='train') #读取训练参数
    
    main(opt)