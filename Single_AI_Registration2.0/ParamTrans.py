'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-02 16:00:38
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-12-04 11:06:30
FilePath: /xushaokang/Single_AI_Registration2.0/ParamTrans.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# 通过上述两组参数，计算合成X-ray图像的变换参数
import sys
from util import get_T
import numpy as np
import random

sys.path.append('../Single_AI_Registration2.0/DRR_modules/')
import RigidMotionModule as rm

def Param_Trans(Ver_Rot_center):
     # 随机产生初始位置参数
    rx_DRR = round(random.uniform(-0.05, 0.05),3)
    ry_DRR = round(random.uniform(-0.05, 0.05),3)
    rz_DRR = round(random.uniform(-0.05, 0.05),3)
    tx_DRR = round(random.uniform(-5, 5),1)
    ty_DRR = round(random.uniform(-5, 5),1)
    tz_DRR = round(random.uniform(-5, 5),1)  
    transform_parameters_DRR = [rx_DRR, ry_DRR, rz_DRR,tx_DRR,ty_DRR, tz_DRR]

    # 随机产生初定位与X-ray之间投影变换参数差值  
    delta_rx =round(random.uniform(-0.15, 0.15),3)
    delta_ry = round(random.uniform(-0.15, 0.15),3)
    delta_rz = round(random.uniform(-0.15, 0.15),3)
    delta_tx = round(random.uniform(-10, 10),1)
    delta_ty = round(random.uniform(-10, 10),1)
    delta_tz = round(random.uniform(-10, 10),1)
    transform_parameters_delta = [delta_rx, delta_ry, delta_rz,delta_tx,delta_ty, delta_tz]

    RT_init = rm.get_rigid_motion_mat_from_euler(transform_parameters_DRR[0], 'x', 
                                                 transform_parameters_DRR[1], 'y', 
                                                 transform_parameters_DRR[2], 'z', 
                                                 transform_parameters_DRR[3], 
                                                 transform_parameters_DRR[4],
                                                 transform_parameters_DRR[5], 
                                                 Ver_Rot_center)
                                                 
    RotCenterCordInit = RT_init[:3, :3]@Ver_Rot_center.T+RT_init[:3, 3]#椎体质心的位置根据初始矩阵做对应变换                                        
    RT_delta = rm.get_rigid_motion_mat_from_euler(transform_parameters_delta[0], 'x', 
                                                  transform_parameters_delta[1], 'y', 
                                                  transform_parameters_delta[2], 'z', 
                                                  transform_parameters_delta[3],
                                                  transform_parameters_delta[4], 
                                                  transform_parameters_delta[5], 
                                                  RotCenterCordInit) 

    RT_X = np.dot(RT_delta, RT_init)  #旋转矩阵变换
    transform_parameters_X = get_T(RT_X, Ver_Rot_center)#根据旋转矩阵以及旋转点计算6个变换参数
    RotCenterCordXray = RT_X[:3, :3]@Ver_Rot_center.T+RT_X[:3, 3]#椎体质心的位置根据目标矩阵做对应变换 
    """ RotCenterCordXray1 = RT_delta[:3, :3]@RotCenterCordInit.T+RT_delta[:3, 3]#椎体质心的位置根据初始矩阵做对应变换
    print(RotCenterCordXray1) 
    exit() """
    return transform_parameters_DRR,transform_parameters_delta,transform_parameters_X,RotCenterCordInit,RotCenterCordXray, RT_X, RT_init