'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-05 16:31:22
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-05 16:39:23
FilePath: /xushaokang/Single_AI_Registration2.0/onnx.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import netron
import torch
from model import Pose_Net

model=Pose_Net()
onnx_path="model.onnx"

torch .onnx.export(model, (torch.randn(1,1,512,512),torch.randn(1,1,512,512)), onnx_path,opset_version=11)
netron.start(onnx_path)