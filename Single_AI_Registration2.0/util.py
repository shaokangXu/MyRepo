'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-02 15:58:52
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-11-01 15:57:08
FilePath: /xushaokang/Single_AI_Registration2.0/util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torchvision import transforms
from PIL import Image
import numpy as np
import itk
import torch
import matplotlib.pyplot as plt
import cv2
import random
import torch
from matplotlib.path import Path
device = torch.device('cuda')


import matplotlib.patches as patches


def point_in_which_bbox(point, bboxs):
    VerBox=[]
    for bbox in bboxs:
            if is_point_in_polygon(point, bbox):
                VerBox = bbox.tolist()
                break  # 找到对应的包围框就停止搜索
    return VerBox

def is_point_in_polygon(point, polygon):
    """
    判断点是否在由四个点定义的多边形（四边形）内
    :param point: (x, y) 标记点的坐标
    :param polygon: 包围框的顶点坐标 [X1, Y1, X2, Y2, X3, Y3, X4, Y4]
    :return: bool 是否在框内
    """
    top=min(polygon[1],polygon[5],polygon[3],polygon[7])
    bottom=max(polygon[1],polygon[5],polygon[3],polygon[7])
    
    if(top<point[1] and point[1]<bottom):
        return True
    else:
        return False
    
def draw_bboxes_with_names(image, name_to_bbox, save_path):
    """
    在图像中绘制包围框并写上对应的名字，然后保存图像
    :param image: 要显示的图像
    :param name_to_bbox: dict 名字到包围框的映射
    :param save_path: 保存图像的路径
    """


    image = image.squeeze().squeeze().cpu().numpy().astype(np.float32)
    image*=255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for name, bbox in name_to_bbox.items():
        # 将bbox的坐标转换为整数
        pts = np.array([
            [int(bbox[0]), int(bbox[1])], 
            [int(bbox[2]), int(bbox[3])], 
            [int(bbox[4]), int(bbox[5])], 
            [int(bbox[6]), int(bbox[7])]
        ])
        
        # 绘制多边形（包围框）
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=1)
        
        # 计算包围框的中心点
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        
        # 在包围框中心写上名字
        cv2.putText(image, name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)
    
    # 保存图像
    
    cv2.imwrite(save_path, image)



def SaveImage(image,name):

    # 将张量转换为 NumPy 数组，并确保数据类型为 uint8
    array = image.squeeze().cpu().numpy() * 255  # 如果数据在 [0, 1] 范围内，需要乘以 255
    array = array.astype(np.uint8)

    # 将 NumPy 数组转换为 PIL 图像
    image = Image.fromarray(array)

    # 保存为图像文件
    image.save(name)


class LinearDecayScheduler:
    def __init__(self, optimizer, start_epoch, total_epochs, initial_lr):
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr

    def step(self, epoch):
        if epoch < self.start_epoch:
            lr = self.initial_lr
        else:
            decay_steps = self.total_epochs - self.start_epoch
            lr = self.initial_lr * (1 - (epoch - self.start_epoch) / decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def Save_train_info(epoch,num_epochs,loss,learning_rates,SaveName):
    # Save information to a txt file
    file_path = 'Output/'+SaveName+'.txt'
    with open(file_path, 'a') as file:
            file.write(f'Epoch {epoch}/{num_epochs}, Average Loss: {loss}, Current Learning Rate: {learning_rates}\n')
    # 绘制折线图
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # 提取每个epoch的信息
        epoch_numbers = []
        average_losses = []

        for line in lines:
            # 假设每行的格式是 "Epoch x/500, Average Loss: y, Current Learning Rate: z"
            if "Epoch" in line and "Average Loss" in line:
                epoch_number = int(line.split('/')[0].split()[-1])
                average_loss = float(line.split('Average Loss:')[1].split(',')[0])
                epoch_numbers.append(epoch_number)
                average_losses.append(average_loss)

        # 绘制损失曲线
        plt.plot(epoch_numbers, average_losses, marker='o', linestyle='-', color='b')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.savefig('Output/'+SaveName+view+'.png')
        plt.close()


def Image_Process(image,image_size):
  #obtain pix data from itk 
  image = itk.array_from_image(image) 
  image=np.flip(image,1)
  
  image = np.squeeze(image)
  image =(image - image.min())/(image.max() - image.min())*255

  image = Image.fromarray(image.astype(np.uint8)).convert("L") 
  
  _transforms = [ transforms.Resize(int(image_size)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5), (0.5)) ]
  """ else:
    image = Image.fromarray(image.astype(np.uint8)).convert("RGB")  #channel 1 to 3
  
    _transforms = [ transforms.Resize(int(image_size)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]"""

  transform = transforms.Compose(_transforms)  
  Img_input=transform(image).unsqueeze(0)
  return Img_input


def get_T(RT_result,center):
    # #calculate poss to matrix
    # RT1 = get_RT(Pose1)
    # RT2 = get_RT(Pose2)
    # #trnsformer
    # RT_result = RT2 * RT1

    # Extract rotation matrix R from RT matrix
    R = RT_result[:3, :3]

    T = RT_result[:3, 3]
    # Calculate Euler angles (alpha, beta, gamma) from rotation matrix R
    rx = np.arctan2(R[2, 1], R[2, 2])
    ry = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    rz = np.arctan2(R[1, 0], R[0, 0])

    translation_relative_to_center = T+ np.dot( R, center)-center



    return [rx,ry,rz,translation_relative_to_center[0],translation_relative_to_center[1],translation_relative_to_center[2]]