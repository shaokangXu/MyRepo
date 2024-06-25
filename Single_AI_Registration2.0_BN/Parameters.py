'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-31 09:08:56
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-06-25 09:57:54
FilePath: /xushaokang/Single_AI_Registration/Parameters.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse

def Read_Parameters(mode):
    parser = argparse.ArgumentParser()
    view="AP"
    if mode=="train": 
        #------------train--------------       
        parser.add_argument('--start_epoch', type=int, default=32, help='starting epoch')
        parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
        parser.add_argument('--decay_epochs', type=int, default=100, help='number of epochs of training')  
        parser.add_argument('--cuda', default= True, help='use GPU computation')
        parser.add_argument('--Pose_Net', type=str, default='checkpoint_model/Pose_Net_Single_'+view+'31.pth', help='AP_CNN checkpoint file')
        parser.add_argument('--batch_size', type=float, default=32, help='batchsize')
        parser.add_argument('--xray_per_ct', type=float, default=32, help='Number of X-rays per CT')
        
        parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate')
        parser.add_argument('--step_size', type=float, default=20, help='step_size of lr_scheduler ')
        parser.add_argument('--gamma', type=float, default=0.8, help='gamma of lr_scheduler ')

        parser.add_argument('--num_classes', type=float, default=6, help='num_classes 3 or 6')
        parser.add_argument('--data_folder', type=str, default='../Single_Dataset/ct_Single_train_others.txt', help='Dataset route')
        parser.add_argument('--load_dict', default=True, help='load the state model weight')

    if mode=="test":
        #-----------test-------------
        parser.add_argument('--cuda', default= True, help='use GPU computation')
        parser.add_argument('--Pose_Net', type=str, default='checkpoint_model/Pose_Net_Single_'+view+'241.pth', help='AP_CNN checkpoint file')
        parser.add_argument('--batch_size', type=float, default=1, help='batchsize')
        parser.add_argument('--xray_per_ct', type=float, default=1, help='Number of X-rays per CT')
        parser.add_argument('--data_folder', type=str, default='../Single_Dataset/ct_Single_test_others.txt', help='Dataset route')
    
    parser.add_argument('--view', type=str, default=view, help='AP or LAT')  
    opt = parser.parse_args()
    if opt.view != "AP"  and opt.view != "LAT":
        print("View only choose AP or LAT, Please choose again!!")
        exit()
    """ if opt.num_classes != 3 and opt.num_classes != 6:
        print("num_classes only choose 3 or 6, Please choose again!!")
        exit()
    print(opt) """
    return opt

def common():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default= True, help='use GPU computation')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--image_size', type=int, default=512, help='image_size')
    parser.add_argument('--AP_generator_A2B', type=str, default='checkpoint_model/AP_X-ray_netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--LAT_generator_A2B', type=str, default='checkpoint_model/LAT_X-ray_netG_A2B.pth', help='A2B generator checkpoint file')
    opt = parser.parse_args()
    return opt