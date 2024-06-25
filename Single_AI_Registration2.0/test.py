'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-13 10:07:01
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-03-04 10:45:28
FilePath: /xushaokang/Single_AI_Registration/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
from torchvision.utils import save_image
import torch
from model import Pose_Net
from torch.utils.data import DataLoader
from datasets import CustomDataset
from  SaveResultData import save_result
from Parameters import Read_Parameters
device = torch.device('cuda')

def main(opt):
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    pose_net = Pose_Net()
    if opt.cuda:
        pose_net.cuda()
    #loda model
    pose_net.load_state_dict(torch.load(opt.Pose_Net))
    
    # Create a DataLoader
    with torch.no_grad():
        pose_net.eval()
        with open(opt.data_folder, 'r') as file:
        # 使用 len() 函数获取CT数量
            lines = file.readlines()      
        for j,line in enumerate(lines):           
            print(line)
            train_loader = DataLoader(CustomDataset(line, opt.xray_per_ct,opt.view), batch_size = opt.batch_size, shuffle=True, pin_memory=True)
            for i, (DRR_init, X_ray,transform_parameters_delta) in enumerate(train_loader):
                start1=time.time()
                DRR_init=DRR_init.to(device)
                X_ray=X_ray.to(device)
                transform_parameters_delta=transform_parameters_delta.to(device)  

                # Enter Network
                T= pose_net(DRR_init,X_ray)   #预测              
                #将回归值调整回原始参数
                T[:,:3]*=5.7    # ((num/10)/3.14)*180
                T[:,-3:]*=5 
                print("Pre:",T)
                transform_parameters_delta[:,:3]*=57
                print("GT:",transform_parameters_delta)
                Time=round(time.time()-start1,3)  #计算配准时间
                save_result(transform_parameters_delta,T,i,Time)  #保存相关输出数据信息(误差、配准时间)
                
                print("Error",torch.round(abs(T-transform_parameters_delta),decimals=3))
                print("配准耗时:",Time,"秒")


                DRR_init=(DRR_init*255 ).to(torch.uint8).cpu().squeeze(0)
                X_ray=(X_ray*255 ).to(torch.uint8).cpu().squeeze(0)
                import torchvision.transforms.functional as TF
                from PIL import Image
                pil_image = TF.to_pil_image(DRR_init)
                pil_image.save("Output/DRR_init"+opt.view+".png")
                pil_image = TF.to_pil_image(X_ray)
                pil_image.save("Output/X_ray"+opt.view+".png")
                


if __name__ == '__main__':
    opt=Read_Parameters(mode='test') #读取测试参数
    main(opt)







