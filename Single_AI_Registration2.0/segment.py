'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-10-19 13:31:31
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-10-15 14:26:18
FilePath: /xushaokang/SingleRegistration/segment.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import random
from torchvision import transforms
device = torch.device('cuda')

def segmentation(imgs,pred_poly,index,view):
    # 创建一个单通道的图像，大小为 (512, 512)

    imgs=normalize(imgs)
    
    # 假设你有四个点的坐标
    if view=="AP":
        point1 = (int(pred_poly[index,0])+10, int(pred_poly[index,1])-10)
        point2 = (int(pred_poly[index,2])+10, int(pred_poly[index,3])+10)
        point3 = (int(pred_poly[index,4])-10, int(pred_poly[index,5])+10)
        point4 = (int(pred_poly[index,6])-10, int(pred_poly[index,7])-10)
    else:
        point1 = (int(pred_poly[index,0])+60, int(pred_poly[index,1])-10)
        point2 = (int(pred_poly[index,2])+60, int(pred_poly[index,3])+10)
        point3 = (int(pred_poly[index,4])-10, int(pred_poly[index,5])+10)
        point4 = (int(pred_poly[index,6])-10, int(pred_poly[index,7])-10)

    
    # 创建一个与图像大小相同的目标图像，初始化为全黑（0）
    # 定义四个点作为多边形的顶点
    pts = np.array([point1, point2, point3, point4], np.int32)
    # 创建一个空白的黑色图像
    mask = np.zeros_like(imgs)
    # 填充框内的区域为白色
    cv2.fillPoly(mask, [pts], (255, 255, 255))       
    # 反转遮罩，将框外的区域变成白色        
    mask = cv2.bitwise_not(mask)
    # 将框内的区域变成黑色
    result = cv2.bitwise_and(imgs, mask)
    # 保存目标图像，其中包含四边形内的区域

    result_image = Image.fromarray(imgs-result.astype(np.uint8)).convert("L")
    _transforms = [transforms.Resize(int(512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))]

    transform = transforms.Compose(_transforms)
    Img_input = transform(result_image).unsqueeze(0)
    return  0.5*(Img_input+1)  


def ExtractedByCoord(source_image,CoordList):
    source_image=normalize(source_image).astype(np.uint8)
    CoordList = [int(num) for num in CoordList] # 将列表中的所有元素转换为整数
    # 找到第一张图像中有内容的纵向最高点和最低点
    top = min(CoordList[1],CoordList[3],CoordList[5],CoordList[7])
    bottom=max(CoordList[1],CoordList[3],CoordList[5],CoordList[7])

    # 找到第一张图像中有内容的横向最左点和最右点
    left = min(CoordList[0],CoordList[2],CoordList[4],CoordList[6])
    right =max(CoordList[0],CoordList[2],CoordList[4],CoordList[6])
    height=bottom-top
    length=right-left
    # 将框扩大(6-10)%
    expandRate= round(random.uniform(0.06,0.10),2)
    ExpandHeight=int(height*expandRate)
    ExpandLength=int(length*expandRate)
    
    if(top-ExpandHeight<=0):
        top=0
    else:
        top=top-ExpandHeight

    if(bottom + ExpandHeight>source_image.shape[0]):
        bottom=source_image.shape[0]
    else:
        bottom=bottom+ExpandHeight

    if(left-ExpandLength<=0):
        left=0
    else:
        left=left-ExpandLength

    if(right+ExpandLength>source_image.shape[1]):
        right=source_image.shape[1]
    else:
        right=right+ExpandLength

    
    # 创建一个与第二张图像相同尺寸的黑色图像
    result_image = np.zeros_like(source_image)

    # 仅在新图像中保留第二张图像中最高点、最低点、最左点和最右点之间的区域内容
    result_image[top:bottom, left:right] = source_image[top:bottom, left:right]

    # 保存新的图像

    result_image = Image.fromarray(result_image.astype(np.uint8)).convert("L")
    _transforms = [transforms.Resize(int(512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))]

    transform = transforms.Compose(_transforms)
    Img_input = transform(result_image).unsqueeze(0)

    return 0.5 * (Img_input + 1)

def extracted(source_image,target_image):
    source_image=normalize(source_image).astype(np.uint8)
    target_image=normalize(target_image).astype(np.uint8)

    """ # 确保两张图像具有相同的尺寸
    if source_image.shape[:2] != target_image.shape[:2]:
        raise ValueError("两张图像的尺寸不一致")

    non_black_pixels = source_image > 8

    result_image = source_image.copy()
    result_image[non_black_pixels] = target_image[non_black_pixels]

    # 保存叠加后的图像
    cv2.imwrite('result_image.jpg', result_image) """
    # 找到第一张图像中有内容的纵向最高点和最低点
    non_black_rows = np.any(source_image > 50, axis=1)
    top = np.argmax(non_black_rows)
    bottom = len(non_black_rows) - np.argmax(non_black_rows[::-1]) -10

    # 创建一个与第二张图像相同尺寸的黑色图像
    result_image = np.zeros_like(target_image)

    # 仅在新图像中保留第二张图像中最高点与最低点之间的区域内容
    result_image[top:bottom, :] = target_image[top:bottom, :]

    result_image = Image.fromarray(result_image.astype(np.uint8)).convert("L")
    _transforms = [ transforms.Resize(int(512)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5), (0.5)) ]

    transform = transforms.Compose(_transforms) 
    Img_input=transform(result_image).unsqueeze(0)
    
    return 0.5*(Img_input+1)

def extracted_retangle(source_image,target_image):
    source_image=normalize(source_image).astype(np.uint8)
    target_image=normalize(target_image).astype(np.uint8)

    # 找到第一张图像中有内容的纵向最高点和最低点
    non_black_rows = np.any(source_image > 50, axis=1)
    top = np.argmax(non_black_rows)
    if top >=10: 
        top=top-10
    bottom = len(non_black_rows) - np.argmax(non_black_rows[::-1]) - 10
    if abs(top-bottom)<70 or abs(top-bottom)>300:
        Flag= False
    else :
        Flag=True


    # 找到第一张图像中有内容的横向最左点和最右点
    non_black_columns = np.any(source_image > 50, axis=0)
    left = np.argmax(non_black_columns)-20
    right = len(non_black_columns) - np.argmax(non_black_columns[::-1]) - 1+15

    
    # 创建一个与第二张图像相同尺寸的黑色图像
    result_image = np.zeros_like(target_image)

    # 仅在新图像中保留第二张图像中最高点、最低点、最左点和最右点之间的区域内容
    result_image[top:bottom, left:right] = target_image[top:bottom, left:right]

    # 保存新的图像

    result_image = Image.fromarray(result_image.astype(np.uint8)).convert("L")
    _transforms = [transforms.Resize(int(512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))]

    transform = transforms.Compose(_transforms)
    Img_input = transform(result_image).unsqueeze(0)

    return 0.5 * (Img_input + 1),Flag


def extracted_DRR(source_image,target_image):
    source_image=normalize(source_image).astype(np.uint8)
    target_image=normalize(target_image).astype(np.uint8)

    # 确保两张图像具有相同的尺寸
    if source_image.shape[:2] != target_image.shape[:2]:
        raise ValueError("两张图像的尺寸不一致")

    non_black_pixels = source_image > 40

    result_image = np.zeros_like(source_image)
    result_image[non_black_pixels] = target_image[non_black_pixels]

    result_image = Image.fromarray(result_image.astype(np.uint8)).convert("L")
    _transforms = [ transforms.Resize(int(512)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5), (0.5)) ]

    transform = transforms.Compose(_transforms) 
    Img_input=transform(result_image).unsqueeze(0)
    
    return 0.5*(Img_input+1)



def normalize(imgs):
    min_value = imgs.min()
    max_value = imgs.max()

    # 将 tensor 缩放到 0 到 255 范围内
    imgs = 255 * (imgs - min_value) / (max_value - min_value)
    imgs = imgs.squeeze(0).cpu().numpy()

    imgs = imgs.squeeze(0)
    return imgs
