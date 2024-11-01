'''
Author: qiuyi.ye qiuyi.ye@maestrosurgical.com
Date: 2024-08-12 18:07:17
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-08-13 13:17:53
FilePath: /xushaokang/Single_AI_Registration2.0_BN3/del.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from model import Pose_Net

# 1. 加载模型
model = Pose_Net()
model.load_state_dict(torch.load('checkpoint_model/Pose_Net_Single_AP52.pth'))
model.denseNet=None


torch.save(model.state_dict(), 'checkpoint_model/Pose_Net_Single_AP52.pth')
