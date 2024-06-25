'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-13 10:43:15
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-02 17:04:19
FilePath: /xushaokang/Single_AI_Registration/Style_trans.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#-------Style transform---------  
#-------------------------------  
import torch
from model import Generator
from Parameters import common

opt=common() #调用参数列表
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

  if view =="AP":
    #CycleGAN transform
    DRR2X_image=(netG_A2B_AP(image).data +1)*0.5
  else:
    #CycleGAN transformF
    DRR2X_image=(netG_A2B_LAT(image).data +1)*0.5
  return DRR2X_image