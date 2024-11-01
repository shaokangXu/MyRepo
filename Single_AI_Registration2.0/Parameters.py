'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-31 09:08:56
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-11-01 16:21:28
FilePath: /xushaokang/Single_AI_Registration/Parameters.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse

def Read_Parameters(mode):
    parser = argparse.ArgumentParser()
    if mode=="train": 
        #------------train-------------- 
        parser.add_argument('--start_epoch', type=int, default=1, help='starting epoch')
        parser.add_argument('--epochs', type=int, default=60, help='number of epochs of training')
        parser.add_argument('--decay_epochs', type=int, default=10, help='number of epochs of training')  
        parser.add_argument('--cuda', default= True, help='use GPU computation')
        parser.add_argument('--Pose_Net', type=str, default='checkpoint_model/Pose_Net_Single25.pth', help='AP_CNN checkpoint file')
        parser.add_argument('--batch_size', type=float, default=1, help='batchsize')
        parser.add_argument('--xray_per_ct', type=float, default=16, help='Number of X-rays per CT')
        parser.add_argument('--modelName', type=str, default="m2", help='')
        
        parser.add_argument('--lr', type=float, default=0.00015, help='initial learning rate')

        parser.add_argument('--data_folder', type=str, default='/home/ps/xushaokang/Datasets/ctd.txt', help='Dataset route')
        parser.add_argument('--load_dict', default=True, help='load the state model weight')
        parser.add_argument('--LossDataName', default="Training_info_", help='Save the loss data')

    if mode=="test":
        #-----------test------------- 
        parser.add_argument('--cuda', default= True, help='use GPU computation')
        parser.add_argument('--Pose_Net', type=str, default='checkpoint_model/Pose_Net_Single21.pth', help='AP_CNN checkpoint file')
        parser.add_argument('--batch_size', type=float, default=1, help='batchsize')
        parser.add_argument('--xray_per_ct', type=float, default=200, help='Number of X-rays per CT')
        parser.add_argument('--data_folder', type=str, default='/home/ps/xushaokang/Datasets/ctd-test.txt', help='Dataset route')
    
    parser.add_argument('--mode', type=str, default="train", help='train or test') 

    opt = parser.parse_args()
    return opt

def common():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default= True, help='use GPU computation')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--image_size', type=int, default=512, help='image_size')
    parser.add_argument('--AP_generator_A2B', type=str, default='checkpoint_model/AP_X-ray_netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--LAT_generator_A2B', type=str, default='checkpoint_model/AP_X-ray_netG_A2B.pth', help='A2B generator checkpoint file')
    opt = parser.parse_args()
    return opt