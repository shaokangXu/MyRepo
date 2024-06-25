import numpy as np
import cv2
import os
import sys
from pathlib import Path
import shutil
"""
image_path = "/workspace/jiangzongkang/dockerCode/spineData/split_data2/images/sunhl-1th-30-Dec-2016-161 A AP__1__0__931.png"
img = cv2.imread(image_path, 0)

# gamma变换
g_imgGamma = np.zeros(img.shape, np.uint8)
cv2.intensity_transform.gammaCorrection(img, g_imgGamma, 130 / 100.0)

print(g_imgGamma.shape)
"""

p = "/workspace/jiangzongkang/dockerCode/spineData/orig_data/autosplit_test.txt"
p = Path(p)  # os-agnostic
f = []  # image files
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

with open(p) as t:
    t = t.read().strip().splitlines()
    parent = str(p.parent) + os.sep
    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path

img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)                     
# print(img_files)

ori_test = "/workspace/jiangzongkang/dockerCode/spineData/ori_test"

for x in img_files:
    print(x)
    shutil.copy(x, os.path.join(ori_test, x.split("/")[-1]))