'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-06-26 17:03:50
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-16 11:35:07
FilePath: /xushaokang/PyTorch-CycleGAN-master-master/datasets.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import glob
import random
import os
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/DRR_LAT' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/X-ray_LAT' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('L'))
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L'))
            #item_B = item_B.resize((item_A.size[0], item_A.size[1]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('L'))
        
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))