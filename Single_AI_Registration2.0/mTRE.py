'''
Author: qiuyi.ye qiuyi.ye@maestrosurgical.com
Date: 2024-11-01 17:04:56
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-11-11 09:06:16
FilePath: /xushaokang/Single_AI_Registration2.0/mTRE.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import numpy as np
import random
sys.path.append('../Single_AI_Registration2.0/DRR_modules/')
import RigidMotionModule as rm
import math
def TRE(RotCenterCordInit,RT_X,RT_init,T,landmarks):
    RT_X = np.array(RT_X.squeeze(0).to('cpu').tolist())
    RT_init = np.array(RT_init.squeeze(0).to('cpu').tolist())
    PreT = T.squeeze(0).to('cpu').tolist()
    RotCenter = np.array(RotCenterCordInit.squeeze(0).to('cpu').tolist())

    
    Pre_RT_delta = rm.get_rigid_motion_mat_from_euler(PreT[0], 'x', 
                                                  PreT[1], 'y', 
                                                  PreT[2], 'z', 
                                                  PreT[3],
                                                  PreT[4], 
                                                  PreT[5], 
                                                  RotCenter)
    pre_RT_X=np.dot(Pre_RT_delta, RT_init)
    landmarks=np.array(landmarks)

    A = pre_RT_X[:3, :3]@landmarks.T+pre_RT_X[:3, 3]
    B = RT_X[:3, :3]@landmarks.T+RT_X[:3, 3]
    #print(A,"---->",B)
    tre_value=math.sqrt(np.sum((A-B)**2))
    return tre_value