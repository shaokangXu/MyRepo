'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-10-27 11:06:53
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-27 11:17:35
FilePath: /xushaokang/AI_Registration4/Style_trans.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#-------Style transform---------  
#-------------------------------  
import torch
from utils import open_image
import argparse
import os
import random
from Wave import WAVE
from models import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default= True, help='use GPU computation')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--AP_generator_A2B', type=str, default='train_model/Spine_netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--LAT_generator_A2B', type=str, default='train_model/LAT_Spine_netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--image_size', type=int, default=512)
opt = parser.parse_args()

netG_A2B_AP = Generator(opt.input_nc, opt.output_nc)
netG_A2B_LAT = Generator(opt.input_nc, opt.output_nc)
if opt.cuda:
    netG_A2B_AP.cuda()
    netG_A2B_LAT.cuda()


# Load state dicts
device = torch.device('cuda')
netG_A2B_AP.load_state_dict(torch.load(opt.AP_generator_A2B))
netG_A2B_LAT.load_state_dict(torch.load(opt.LAT_generator_A2B))
# Set model's test mode
netG_A2B_AP.eval()
netG_A2B_LAT.eval()
#--------------Style-----------------
#random obtain X-ray 
def Style_trans(image,view):

  # WAVE-CT transform
  X_ray_path= '../Datasets/'
  file_list_AP = os.listdir(X_ray_path+"AP_X-ray/")
  file_list_LAT = os.listdir(X_ray_path+"LAT_X-ray/")
  if view =="AP":
    #CycleGAN transform
    DRR2X_image=(netG_A2B_AP(image).data +1)*0.5
    style = open_image(X_ray_path+"AP_X-ray/"+random.choice(file_list_AP), opt.image_size).to(device)
    DRR2X_image = WAVE(DRR2X_image, style)
    
  else:
    #CycleGAN transformF
    DRR2X_image=(netG_A2B_LAT(image).data +1)*0.5
    style = open_image(X_ray_path+"LAT_X-ray/"+random.choice(file_list_LAT), opt.image_size).to(device)
    DRR2X_image = WAVE(DRR2X_image, style)

  return DRR2X_image