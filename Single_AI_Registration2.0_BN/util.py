'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-02 15:58:52
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-06-25 09:54:49
FilePath: /xushaokang/Single_AI_Registration2.0/util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torchvision import transforms
from PIL import Image
import numpy as np
import itk
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda')

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

def Save_train_info(epoch,num_epochs,loss,learning_rates,view):
    # Save information to a txt file
    file_path = 'Output/training_info_'+view+'.txt'
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
        plt.savefig('Output/training_loss_plot'+view+'.png')
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