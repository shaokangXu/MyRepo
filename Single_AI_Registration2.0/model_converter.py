'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-19 09:30:03
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-09-06 17:09:14
FilePath: /xushaokang/Single_AI_Registration2.0_fast2/model_converter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os, cv2
import torch
from model import Pose_Net
import numpy as np
import argparse



def format_onnx_name(opt):
    onnx_filename = opt.weights + '.onnx'
    return onnx_filename


def format_trt_name(opt):
    trt_filename = opt.weights + '.trt'
    return trt_filename


def main(opt):
    if opt.weights == '':
        exit('load model required!')

    # Step 1: Build torch model and load weights
    print('Loading model...')
    """ if opt.model == 'G':
        model = Generator(3, 3)
    else:
        model = Discriminator(3) """
    model=Pose_Net()
    model.load_state_dict(torch.load(opt.weights))
    dummy_input1 = torch.randn(1, 1, 512, 512).to('cuda')
    dummy_input2 = torch.randn(1, 1, 512, 512).to('cuda')
    img_tensor=(dummy_input1, dummy_input2)
    model.to('cuda')
    model.eval()

    # Step 2: Convert torch model to ONNX
    onnx_filename = format_onnx_name(opt)
    torch.onnx.export(model,
                      img_tensor,
                      onnx_filename,
                      input_names=['input1','input2'],
                      output_names=['output'],
                      verbose=False,
                      opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL)

    """ try:
        onnx.checker.check_model(onnx_filename)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s" % e)
    else:
        print("The model is valid!") """

    model.to('cpu')
    del model

    # Step 3: Convert ONNX to tensorRT
    trt_filename = format_trt_name(opt)
    os.system('trtexec --onnx={} --saveEngine={} --workspace=8192'.format(onnx_filename, trt_filename))
    # convert_onnx_to_trt(onnx_filename, trt_filename)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='checkpoint_model/Pose_Net_Single_AP55.pth',
                        help='pth file path for Drr2Xray model')

    opt = parser.parse_args()
    main(opt)
