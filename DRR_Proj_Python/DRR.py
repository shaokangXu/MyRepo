'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-29 15:23:33
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-05-11 11:52:36
FilePath: /xushaokang/MaestroAlgoImgRegisDL/DRR_Proj_Python/DRR.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import time
import glob
import os
import itk
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import random
sys.path.append('../DRR_Proj_Python/modules/')
import ProjectorsModule as pm
import ReadWriteImageModule as rw



projector_info = {
    'Name': 'SiddonGpu',
    'threadsPerBlock_x': 16,
    'threadsPerBlock_y': 16,
    'threadsPerBlock_z': 1,
    'DRRsize': 1024,
    'DRRspacing': 0.209,
    'DRR_pp': 0.0,
    'focal_lenght': 1011.0
}

PixelType = itk.F
Dimension = 3
ScalarType = itk.D
view="AP"          #Proj view
thresholds=0      #DRR


#Build DRR proj coordinate
#批量投影
""" folder_path = '/home/ps/xushaokang/Single_Dataset/CT'
image_files = sorted(glob.glob(os.path.join(folder_path, '*.gz')))
for i, image_file in enumerate(image_files):
    projector = pm.projector_factory(projector_info, image_file, PixelType, Dimension, ScalarType, view)
    #Perform spatial transformations on CTs and calculate projection
    for j in range(13):
        rx_DRR = round(random.uniform(-0.2, 0.2), 3)
        ry_DRR = round(random.uniform(-0.2, 0.2), 3)
        rz_DRR = round(random.uniform(-0.2, 0.2), 3)
        tx_DRR = round(random.uniform(-3, 3), 2)
        ty_DRR = round(random.uniform(-3, 3), 2)
        tz_DRR = round(random.uniform(-3, 3), 2)
        transform_parameters_DRR = [rx_DRR, ry_DRR, rz_DRR, tx_DRR, ty_DRR, tz_DRR]
        thresholds=random.randint(-30,10)  #随机投影阈值
        drr_image = projector.compute(transform_parameters_DRR,thresholds)   #Proj
        print(i)
        image_type = itk.Image[PixelType, 3]

        # #Save DRR iamge
        name = os.path.splitext(os.path.splitext(os.path.basename(image_file))[0])[0]
        image_file_name="/home/ps/xushaokang/Single_Dataset/DRR/AAALAT"+name+str(j)
        rw.ImageWriter(drr_image,image_type, image_file_name)
        exit()
    projector.delete() """


#单个CT
ct_path="xx.nii.gz"
movingImageFileName = ct_path
projector = pm.projector_factory(projector_info, movingImageFileName, PixelType, Dimension, ScalarType, view)
#Perform spatial transformations on CTs and calculate projection
transform_parameters = [0,0.1,0,0,0,0]  # Six param

drr_image = projector.compute(transform_parameters,thresholds)   #Proj
image_type = itk.Image[PixelType, 3]

# # #Save DRR iamge

rw.ImageWriter(drr_image,image_type, view)
projector.delete()

def Image_Process(image, image_size):
    #obtain pix data from itk
    image = itk.array_from_image(image)
    image = np.flip(image, 1)
    image = np.squeeze(image)
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = Image.fromarray(image.astype(np.uint8)).convert("L")  # channel 1 to 3

    _transforms = [transforms.Resize(int(image_size)),
                   transforms.ToTensor()
                    ]
#
    transform = transforms.Compose(_transforms)
    Img_input = transform(image).squeeze(0)
    print(Img_input.shape)
    return Img_input

test1=Image_Process(drr_image,1000)
save_image(test1, 'Reg2.png')
