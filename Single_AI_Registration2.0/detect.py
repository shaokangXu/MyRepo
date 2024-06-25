'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-10-18 13:42:43
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-30 10:59:57
FilePath: /xushaokang/SingleRegistration/detect.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms


#---------Detect--------------
weights='../SingleRegistration/checkpoint_model/best.pt' # model.pt path(s)

imgsz=(512,512)  # inference size (height, width)
conf_thres=0.5 # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image

view_img=False # show results
save_txt=False # save results to *.txt
save_conf=False # save confidences in --save-txt labels
save_crop=False  # save cropped prediction boxes
nosave=False # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
update=False  # update all models
name='exp'  # save results to project/name
half=False # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference
line_thickness=3  # bounding box thickness (pixels)
source='Output'  # file/dir/URL/glob, 0 for webcam
hide_labels=False  # hide labels
hide_conf=False  # hide confidences

# Load model
device = torch.device('cuda')
model = DetectMultiBackend(weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size
source = str(source)
save_img = not nosave and not source.endswith('.txt')  # save inference images

# Half
half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt or jit:
    model.model.half() if half else model.model.float()

model.warmup(imgsz=(1, 1, *imgsz), half=half)  # warmup

#------------     -----------

def detect(imgs,view):
    pred = model(imgs, augment=augment, visualize=visualize)

    # NMS
    # pred: list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
    pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
    

    # 获取按每行第二个元素排序的索引
    sorted_indices = torch.argsort(pred_poly[:, 1])
    # 使用索引对张量进行排序
    pred_poly = pred_poly[sorted_indices]

    
    min_value = imgs.min()
    max_value = imgs.max()

    # 将 tensor 缩放到 0 到 255 范围内
    imgs = 255 * (imgs - min_value) / (max_value - min_value)
    image = imgs.squeeze(0).cpu().numpy()
    image = image.squeeze(0)
    pred_poly=pred_poly.cpu()

    #循环为每个脊椎画上框
    for i in range(pred_poly.shape[0]):

        if view=="AP":
            point1 = (int(pred_poly[i,0])+10, int(pred_poly[i,1])-10)
            point2 = (int(pred_poly[i,2])+10, int(pred_poly[i,3])+10)
            point3 = (int(pred_poly[i,4])-10, int(pred_poly[i,5])+10)
            point4 = (int(pred_poly[i,6])-10, int(pred_poly[i,7])-10)
        else:
            point1 = (int(pred_poly[i,0])+60, int(pred_poly[i,1])-10)
            point2 = (int(pred_poly[i,2])+60, int(pred_poly[i,3])+10)
            point3 = (int(pred_poly[i,4])-10, int(pred_poly[i,5])+10)
            point4 = (int(pred_poly[i,6])-10, int(pred_poly[i,7])-10)

        # 创建一个包含四个点的列表
        polygon = [point1, point2, point3, point4]
        
    
        # 在output_image上绘制边框
        cv2.polylines(image, [np.array(polygon)], isClosed=True, color=255, thickness=2)


    # 保存绘制了框的图像
    from PIL import Image
    image = Image.fromarray(image.astype(np.uint8))
    image.save("out_"+view+".png") 
    return  pred_poly
