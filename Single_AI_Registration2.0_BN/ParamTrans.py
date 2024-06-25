'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-02 16:00:38
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-05-07 17:30:28
FilePath: /xushaokang/Single_AI_Registration2.0/ParamTrans.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# 通过上述两组参数，计算合成X-ray图像的变换参数
import sys
from util import get_T
import numpy as np
import random
sys.path.append('../Single_AI_Registration2.0_BN/DRR_modules/')
import RigidMotionModule as rm

def Param_Trans(Rot_center):
    rx_DRR = round(random.uniform(-0.05, 0.05),3)
    ry_DRR = round(random.uniform(-0.05, 0.05),3)
    rz_DRR = round(random.uniform(-0.05, 0.05),3)
    tx_DRR = round(random.uniform(-8, 8),1)
    ty_DRR = round(random.uniform(-8, 8),1)
    tz_DRR = round(random.uniform(-8, 8),1)  
    """ rx_DRR = 0
    ry_DRR = 0
    rz_DRR = 0
    tx_DRR = 0
    ty_DRR = 0
    tz_DRR = 0 """
    transform_parameters_DRR = [rx_DRR, ry_DRR, rz_DRR,tx_DRR,ty_DRR, tz_DRR]

    # 随机产生初定位与X-ray之间投影变换参数差值  
    delta_rx =round(random.uniform(-0.1, 0.1),3)
    delta_ry = round(random.uniform(-0.1, 0.1),3)
    delta_rz = round(random.uniform(-0.1, 0.1),3)
    delta_tx = round(random.uniform(-5, 5),1)
    delta_ty = round(random.uniform(-5, 5),1)
    delta_tz = round(random.uniform(-5, 5),1)
    """ delta_rx = 0
    delta_ry = 0
    delta_rz = 1.57
    delta_tx = 0
    delta_ty = 0
    delta_tz = 0 """
    transform_parameters_delta = [delta_rx, delta_ry, delta_rz,delta_tx,delta_ty, delta_tz]

    RT_init = rm.get_rigid_motion_mat_from_euler(transform_parameters_DRR[0], 'x', 
                                                 transform_parameters_DRR[1], 'y', 
                                                 transform_parameters_DRR[2], 'z', 
                                                 transform_parameters_DRR[3], 
                                                 transform_parameters_DRR[4],
                                                 transform_parameters_DRR[5], 
                                                 Rot_center)
    RT_delta = rm.get_rigid_motion_mat_from_euler(transform_parameters_delta[0], 'x', 
                                                  transform_parameters_delta[1], 'y', 
                                                  transform_parameters_delta[2], 'z', 
                                                  transform_parameters_delta[3],
                                                  transform_parameters_delta[4], 
                                                  transform_parameters_delta[5], 
                                                  Rot_center)

    RT_X = np.dot(RT_delta, RT_init)  #旋转矩阵变换
    transform_parameters_X = get_T(RT_X, Rot_center)#根据旋转矩阵以及旋转点计算6个变换参数

    return transform_parameters_DRR,transform_parameters_delta,transform_parameters_X