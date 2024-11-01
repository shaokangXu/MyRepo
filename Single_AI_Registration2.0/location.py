'''
Author: qiuyi.ye qiuyi.ye@maestrosurgical.com
Date: 2024-08-20 13:58:20
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-10-18 15:19:24
FilePath: /xushaokang/Single_AI_Registration2.0_BN4/location.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ClassifyNetwork import SCNet2D
import tensorflow as tf
from collections import OrderedDict
from utils.landmark.heatmap_test import HeatmapTest
import numpy as np
import random
import cv2
from PIL import Image
import os
import pickle
import math
from utils.landmark.spine_postprocessing_graph import SpinePostprocessingGraph

network=SCNet2D
num_landmarks = 25
data_format = 'channels_first'
# 获取所有可用的 GPU

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # 设置 GPU 内存增长为按需分配
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
class dotdict(dict):

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
config=dotdict(
        num_filters_base=128,
        activation='lrelu',
        spatial_downsample=16,
        local_activation='none',
        spatial_activation='tanh',
        num_levels=4,
        spacing=0.418,
    )
network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              spatial_downsample=config.spatial_downsample,
                                              local_activation=config.local_activation,
                                              spatial_activation=config.spatial_activation,
                                              num_levels=config.num_levels,
                                              data_format=data_format)
model = network(num_labels=num_landmarks, **network_parameters)
checkpoint = tf.train.Checkpoint(model=model)
status=checkpoint.restore("../Single_AI_Registration2.0/checkpoint_model/ckpt-1000000")
status.expect_partial()


units_distances_pickle = os.path.join('../Single_AI_Registration2.0/checkpoint_model/', "units_distances.pickle")
possible_successors_pickle = os.path.join('../Single_AI_Registration2.0/checkpoint_model/', "possible_successors.pickle")
with open(possible_successors_pickle, 'rb') as f:
    possible_successors = pickle.load(f)
with open(units_distances_pickle, 'rb') as f:
    offset_distances, offsets_mean, distances_mean, distances_std = pickle.load(f)

spine_postprocessing = SpinePostprocessingGraph(num_landmarks=25,
                                                possible_successors=possible_successors,
                                                offsets_mean=offsets_mean,
                                                distances_mean=distances_mean,
                                                distances_std=distances_std,
                                                bias=2.0,
                                                l=0.2)

SpineDict={
    1:"C1",2:"C2",3:"C3",4:"C4",5:"C5",6:"C6",7:"C7",
    8:"T1",9:"T2",10:"T3",11:"T4",12:"T5",13:"T6",14:"T7",15:"T8",16:"T9",17:"T10",18:"T11",19:"T12",
    20:"L1",21:"L2",22:"L3",23:"L4",24:"L5"
}
heatmap_maxima = HeatmapTest(0,
                                False,
                                return_multiple_maxima=True,
                                min_max_value=0.1,
                                smoothing_sigma=2.0)
def location(image):
    image = image.squeeze().squeeze().cpu().numpy()
    image*=255
    image = Image.fromarray(image.astype(np.uint8))
    image = image.resize((214, 214))
    image = np.array(image).astype(np.float32)
    # 将 NumPy 数组转换为 PIL 图像 
    image/=255   
    image1 = np.pad(image, pad_width=((0, 10), (0, 10)), mode='constant', constant_values=0)
    
    image=np.expand_dims(image1, axis=0)
    image=np.expand_dims(image, axis=0)

    result = model(image, training=False)
    prediction = np.mean(result, axis=0)
    prediction = np.squeeze(result, axis=0)
    local_maxima_landmarks = heatmap_maxima.get_landmarks(prediction)

    #判断定位结果是否为空
    if any(local_maxima_landmarks):
        curr_landmarks = spine_postprocessing.solve_local_heatmap_maxima(local_maxima_landmarks)
    else:
        return False
    
    SpineLocalDict={}
    
    for j in range(25):
        point = [0,0]
        if(math.isnan(curr_landmarks[j].coords[0])):
            continue
        else:
            
            point[0]=int(curr_landmarks[j].coords[0]/0.418)
            point[1]=int(curr_landmarks[j].coords[1]/0.418)
            point = tuple(point)
            SpineLocalDict[SpineDict[j+1]]=point

    return SpineLocalDict

def DrawLocationImage(image,Dict,name):
    # 设置字体和颜色
    image = image.squeeze().squeeze().cpu().numpy().astype(np.float32)
    image*=255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)  # 绿色
    thickness = 1
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 遍历字典，在图像上标记点和对应的key
    for key, point in Dict.items():
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # 绘制点
        cv2.circle(image, point, 5, color, -1)
        
        # 在点旁边显示字符串
        cv2.putText(image, key, (point[0] + 10, point[1]), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # 保存并显示标记后的图像
    
    cv2.imwrite("Output/"+name, image)





import torch
from matplotlib.path import Path

def is_point_in_polygon(point, polygon):
    """
    判断点是否在由四个点定义的多边形（四边形）内
    :param point: (x, y) 标记点的坐标
    :param polygon: 包围框的顶点坐标 [X1, Y1, X2, Y2, X3, Y3, X4, Y4]
    :return: bool 是否在框内
    """
    path = Path([(polygon[0], polygon[1]), (polygon[2], polygon[3]), 
                 (polygon[4], polygon[5]), (polygon[6], polygon[7])])
    return path.contains_point(point)

def map_names_to_bboxes(draw_location, drr_ap_location):
    """
    将名字映射到包围框，形成新字典
    :param draw_location: dict 名字到标记点坐标的映射
    :param drr_ap_location: tensor 包围框列表
    :return: dict 名字到包围框的映射
    """
    name_to_bbox = {}
    
    for name, point in draw_location.items():
        for bbox in drr_ap_location:
            if is_point_in_polygon(point, bbox):
                name_to_bbox[name] = bbox.tolist()
                break  # 找到对应的包围框就停止搜索
    
    return name_to_bbox
