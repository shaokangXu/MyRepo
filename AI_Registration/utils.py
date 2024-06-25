from torchvision import transforms
from PIL import Image
import numpy as np
import os 
import pydicom
import itk
import torch

device = torch.device('cuda')





def Image_Process(image,image_size):

  #obtain pix data from itk 
  image = itk.array_from_image(image) 
  image=np.flip(image,1)
  
  image = np.squeeze(image)
  image =(image - image.min())/(image.max() - image.min())*255
  image = Image.fromarray(image.astype(np.uint8)).convert("RGB")  #channel 1 to 3
  
  _transforms = [ transforms.Resize(int(image_size)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

  transform = transforms.Compose(_transforms) 
  Img_input=transform(image).unsqueeze(0)
  return Img_input
  

def open_image(image_path, image_size=None):
    _, ext = os.path.splitext(image_path)
    if ext == '.gz':
        print("ccc")
        img = nib.load(image_path)
        data = img.get_fdata()
        data = np.squeeze(data)
        data = np.clip(data, 0, 255).astype(np.uint8)
        img = Image.fromarray(data)
        img = img.transpose(Image.ROTATE_90).convert('RGB')
    elif ext == '.png':

        img = Image.open(image_path).convert('RGB')
    else:
        ds = pydicom.dcmread(image_path)
        data = ds.pixel_array
        data_min = np.min(data)
        data_max = np.max(data)
        data = (data - data_min) * (255 / (data_max - data_min))
        data = np.clip(data, 0, 255).astype(np.uint8)
        img = Image.fromarray(data).convert('RGB')



    _transforms = []
    if image_size is not None:
        img = transforms.Resize(image_size)(img)
    w, h = img.size
    _transforms.append(transforms.CenterCrop((h // 16 * 16, w // 16 * 16)))
    _transforms.append(transforms.ToTensor())
    transform = transforms.Compose(_transforms)

    #vutils.save_image(transform(img), 'test.jpg', normalize=True)

    return transform(img).unsqueeze(0)
    
    
def adain(content_features, style_features):
    # [64, 180, 528][64, 360, 496]
    style_mean = style_features.mean((2, 3), keepdim=True)
    style_std = style_features.std((2, 3), keepdim=True) + 1e-6
    content_mean = content_features.mean((2, 3), keepdim=True)
    content_std = content_features.std((2, 3), keepdim=True) + 1e-6
    target_feature = style_std * (content_features - content_mean) / content_std + style_mean
    return target_feature


def get_RT(Pose):
    rot1=np.array(([1,0,0],[0, np.cos(Pose[0]), -np.sin(Pose[0])],[0, np.sin(Pose[0]), np.cos(Pose[0])]))
    rot2=np.array(([np.cos(Pose[1]),0,np.sin(Pose[1])],[0, 1, 0],[-np.sin(Pose[1]), 0, np.cos(Pose[1])]))
    rot3=np.array(([np.cos(Pose[2]),-np.sin(Pose[2]),0],[np.sin(Pose[2]), np.cos(Pose[2]), 0],[0, 0, 1]))

    rigid_motion_mat = np.eye(4)
    rigid_motion_mat[:3, :3] = rot3 @ rot2 @ rot1
    rigid_motion_mat[:3, 3] = np.array([Pose[3], Pose[4], Pose[5]])

    return rigid_motion_mat

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