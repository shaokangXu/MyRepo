'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-08-23 13:04:51
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-09-11 14:26:20
FilePath: /xushaokang/test_AI_Registration/chessboard.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread("Output/Reg_LAT00.png")
image2 = cv2.imread("Output/X_ray_LAT00.png")

image3 = cv2.imread("Output/Reg_AP00.png")
image4 = cv2.imread("Output/X_ray_AP00.png")


#assert image1.shape == image2.shape, "The images should have the same size."


rows = 8
cols = 8

def blend(image1,image2,name):
    block_size = (image1.shape[0] // rows, image1.shape[1] // cols)


    chessboard = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)


    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                chessboard[row*block_size[0]:(row+1)*block_size[0], col*block_size[1]:(col+1)*block_size[1]] = image1[row*block_size[0]:(row+1)*block_size[0], col*block_size[1]:(col+1)*block_size[1]]
            else:
                chessboard[row*block_size[0]:(row+1)*block_size[0], col*block_size[1]:(col+1)*block_size[1]] = image2[row*block_size[0]:(row+1)*block_size[0], col*block_size[1]:(col+1)*block_size[1]]

    # 显示棋盘格图像
    cv2.imwrite('Output/chessboard'+name+'.png', chessboard)
blend(image1,image2,'LAT')
blend(image3,image4,'AP')