import os
import json
import random
import math
import sys

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageOps
from glob import glob
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
from einops import rearrange, repeat
import torch.nn.functional as F
from collections import defaultdict, OrderedDict
# Local application/library specific imports
sys.path.append('.')
from dataset.utils import yaw_to_quaternion, quaternion_to_yaw, color_to_rgb, draw_box_3d, get_color, quaternion_multiply
from dataset.nuscenes_dataset_with_path import NuscDataset

device = torch.device("cuda")

if __name__ == "__main__":
    version = "trainval"
    dataroot = "/disk1/jinhua.zjh/DATA/nuscenes"
    final_dim = (256, 384)

    data_aug_conf = {
        "final_dim": final_dim,
        "cams": [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
        ],
        "Ncams": 6,
    }
    nusc = NuScenes(
        version="v1.0-{}".format(version),
        # dataroot=os.path.join(dataroot, version),
        dataroot=os.path.join(dataroot),
        verbose=False,
    )
    traindata = NuscDataset(version, dataroot, is_train=True,
                         data_aug_conf=data_aug_conf, max_boxes_per_sample=80)
    valdata = NuscDataset(version, dataroot, is_train=False,
                         data_aug_conf=data_aug_conf, max_boxes_per_sample=80)
    

    print(f"size of sample train data: {len(traindata)}, size of sample val data: {len(valdata)}")

    # save_image_path = valdata[10]["save_image_path"]
    # print(save_image_path)
    
    # trainloader = torch.utils.data.DataLoader(traindata, batch_size=6,
    #                                           shuffle=False, num_workers=8,
    #                                           pin_memory = False)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=2,
                                              shuffle=False, num_workers=8,
                                              pin_memory = False)

    
    base_path = os.path.join('samples_real_256x384')
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    cameras =  ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

    for batch_index, batch in enumerate(tqdm(valloader)):
        image = batch["image"]
        save_image_path = batch["save_image_path"]
        b, n, c, h, w = image.shape
        for i in range(b):
            for j in range(n):
                img = TF.to_pil_image(image[i][j] * 0.5 + 0.5)
                        
                if n > 1:
                    cam = cameras[j]
                    img_path = save_image_path[cam][i]
                else:
                    img_path = save_image_path[j][i]

                save_path = os.path.join(base_path, img_path.replace('jpg', "png"))
                par_save_path = os.path.dirname(save_path)  

                if not os.path.exists(par_save_path):
                    os.makedirs(par_save_path, exist_ok=True)  
                img.save(save_path)
    
    


