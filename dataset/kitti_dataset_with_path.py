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

from dataset.utils import yaw_to_quaternion, quaternion_to_yaw, color_to_rgb, draw_box_3d, get_color, quaternion_multiply, crop_back_to_original

device = torch.device("cuda")

class KittiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataroot,
        is_train,
        data_aug_conf,
        max_boxes_per_sample=0,
        random_crop = False,
        random_flip = True,
    ):
        self.dataroot = dataroot
        self.is_train = is_train
       
        self.data_aug_conf = data_aug_conf
        self.random_flip = random_flip
        self.pil_to_tensor = transforms.PILToTensor()
        self.max_boxes = max_boxes_per_sample
        # self.scenes = self.get_scenes()

        # if is_train:
        self.images_dir = os.path.join(dataroot, "training", 'image_2')
        self.labels_dir = os.path.join(dataroot, "training" , 'label_2')
        self.calib_dir = os.path.join(dataroot, "training", 'calib')

        self.train_val_file = os.path.join(dataroot, "ImageSets")
        
        # Load train files
        self.image_files = []
        if is_train:    
            train_file_path = os.path.join(self.train_val_file, "train.txt")
            with open(train_file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    self.image_files.append(line + '.png')
        else:
            val_file_path = os.path.join(self.train_val_file, "val.txt")
            with open(val_file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    self.image_files.append(line + '.png')
 

    def total_images(self):
        return len(self)
    
    def pad_image(self, image):
        img = np.array(image)
        h, w, c = img.shape
        ret_img = np.zeros((self.data_aug_conf["final_dim"][0], self.data_aug_conf["final_dim"][1], c))
        pad_y = (self.data_aug_conf["final_dim"][0] - h) // 2
        pad_x = (self.data_aug_conf["final_dim"][1] - w) // 2

        ret_img[pad_y: pad_y + h, pad_x: pad_x + w] = img
        pad_size = np.array([pad_x, pad_y])
        return Image.fromarray(ret_img.astype(np.uint8)), pad_size


    def get_camera_info(self, idx, cams):
        save_image_path = []
        final_dim = [self.data_aug_conf["final_dim"][0], self.data_aug_conf["final_dim"][1]]
        
        box_name = ["" for _ in range(self.max_boxes)]
        box_mask = torch.zeros(self.max_boxes)
        box_corner_3d = torch.zeros(self.max_boxes, 16)

        perl_box_masking_maps = torch.zeros(self.max_boxes, final_dim[0], final_dim[1])
        
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)

        save_image_path.append(image_file)

        label_path = os.path.join(self.labels_dir, image_file).replace('.png', '.txt')
        calib_path = os.path.join(self.calib_dir, image_file).replace('.png', '.txt')

        image = Image.open(image_path).convert('RGB')
        img_size = (image.size[1], image.size[0])

        image, pad_size = self.pad_image(image)
        # print(f"pad size: {pad_size}, ori img size: {img_size}, pad_img {image.size}")

        # Load calibration matrix P2 (3x4)
        with open(calib_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('P2'):
                    values = line.split()[1:]
                    p2_matrix = np.array([float(val) for val in values]).reshape(3, 4)
                    break

        # Load bounding boxes
        with open(label_path, 'r') as file:
            lines = file.readlines()
            annotation_index = 0
            for line in lines:
                values = line.split()
                
                instance_name = values[0]
                if instance_name == 'DontCare':
                    continue
                # Convert string values into float
                h, w, l, x, y, z, yaw = [float(val) for val in values[8:]]
                # Compute rotational matrix
                R = np.array([
                    [np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])
                # 3D bounding box corners

                x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
                y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
                z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])

                indices = [5, 4, 0, 1, 6, 7, 3, 2]
                x_reordered = x_corners[indices]
                y_reordered = y_corners[indices]
                z_reordered = z_corners[indices]

                # corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
                corners_3d = np.dot(R, np.vstack([x_reordered, y_reordered, z_reordered]))
                corners_3d[0, :] += x
                corners_3d[1, :] += y
                corners_3d[2, :] += z


                if np.any(corners_3d[2,:] < 0.1):
                    continue

                # Add ones to make it [3x4] so we can multiply with p2_matrix
                ones = np.ones((1, 8))
                corners_3d_homogeneous = np.vstack((corners_3d, ones))

                # print("P", p2_matrix)?

                # Project the 3D bounding box to the image plane
                corner_coords = np.dot(p2_matrix, corners_3d_homogeneous)
                corner_coords[0, :] /= corner_coords[2, :]
                corner_coords[1, :] /= corner_coords[2, :]

                """
                qs: (2,8) array of vertices for the 3d box in following order:
                    7 -------- 6
                /|         /|
                4 -------- 5 .
                | |        | |
                . 3 -------- 2
                |/    /     |/
                0 -------- 1

                # """
                # dx = (img_size[1] / 2 - crop_size[1] / 2)
                # dy = (img_size[0] / 2 - crop_size[0] / 2)

                bbox_3d = corner_coords[:2, :].copy()

                # bbox_3d[0, :] = (bbox_3d[0, :] - dx)  / crop_size[1]
                # bbox_3d[1, :] = (bbox_3d[1, :] - dy) / crop_size[0]
                bbox_3d[0, :] = (bbox_3d[0, :] + pad_size[0]) / final_dim[1]
                bbox_3d[1, :] = (bbox_3d[1, :] + pad_size[1]) / final_dim[0]


                # if min(bbox_3d[0, :]) <= 0 or min(bbox_3d[1, :]) <= 0 or max(bbox_3d[0, :]) >= crop_size[1] or max(bbox_3d[1, :]) >= crop_size[0]:
                #     continue

                temp_bbox_3d = bbox_3d.copy()
                temp_bbox_3d[0, :] = temp_bbox_3d[0, :] * final_dim[1]
                temp_bbox_3d[1, :] = temp_bbox_3d[1, :] * final_dim[0]

                temp_bbox_3d = temp_bbox_3d.astype(np.int32)# 2*8

                   
                # coordinates to masks
                perl_box_masking_map = np.zeros((final_dim[0], final_dim[1]), dtype=np.uint8) # [H,W]

                box_faces = [
                    [0, 1, 5, 4],
                    [1, 2, 6, 5],
                    [2, 3, 7, 6],
                    [3, 0, 4, 7],
                    [4, 5, 6, 7],
                    [0, 1, 2, 3]]

                for face in box_faces:
                    pts = np.array([temp_bbox_3d[:, i] for i in face], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(perl_box_masking_map, [pts], 1)
             

                perl_box_masking_map = np.where(perl_box_masking_map > 0, 1, 0)
                perl_box_masking_maps[annotation_index, :, :] = torch.tensor(perl_box_masking_map, dtype=torch.float32)

                box_name[annotation_index] = instance_name
                box_mask[annotation_index] = 1
                box_corner_3d[annotation_index, :] = torch.tensor(bbox_3d.T[:,:2].reshape(-1))
                annotation_index += 1
        

        image = (self.pil_to_tensor(image).float() / 255 - 0.5) / 0.5
        img_size = torch.tensor(img_size, dtype=torch.int64)

        image = image[None, :, :, :]
        box_mask = box_mask[None, :]
        box_corner_3d = box_corner_3d[None, :, :]
        box_name = [box_name]

        perl_box_masking_maps = perl_box_masking_maps[None, :, :]
        img_size = img_size[None,:]


        return (
            image,
            box_name,
            box_mask,
            box_corner_3d,
            perl_box_masking_maps,
            save_image_path,
            img_size,
        )
  
    def choose_cams(self):
        if self.data_aug_conf["Ncams"] < len(
            self.data_aug_conf["cams"]
        ):
            cams = [self.data_aug_conf["cams"][0]]
        else:
            cams = self.data_aug_conf["cams"]
        return cams
    
    def vis_getitem_data(self, index=None, out=None, return_tensor=False, name="res.jpg", print_scene_description=False):
        img = torchvision.transforms.functional.to_pil_image( out["image"]*0.5+0.5 )
        canvas = torchvision.transforms.functional.to_pil_image( torch.ones_like(out["image"]) )
        W, H = img.size

        if print_scene_description:
            scene_description = out["scene_description"]
            print(f"scene_description: {scene_description}")

        boxes_3d = []
        for box in out["box"]:   
            box = rearrange(box, '(n c)-> n c', n = 8, c = 2)
            boxes = []
            for i in range(8):
                x, y = box[i]
                boxes.append( torch.tensor([float(x*W), float(y*H)]) )
            boxes = torch.stack(boxes)
            boxes_3d.append(boxes)
        boxes_3d = torch.stack(boxes_3d)
        img = draw_box_3d(img, boxes_3d)
        
        if return_tensor:
            return  torchvision.transforms.functional.to_tensor(img)
        else:
            img.save(name) 


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        cams = self.choose_cams()

        out = {}
        (   image,
            box_name,
            box_mask,
            box_corner_3d,
            perl_box_masking_map,
            save_image_path,
            img_size,
        ) = self.get_camera_info(index, cams)
        

        out["image"] = image
        out["box_name"] = box_name
        out["box_mask"] = box_mask
        out['box'] = box_corner_3d

        out["perl_box_masking_map"] = perl_box_masking_map
        out["scene_description"] = "Realistic autonomous driving scenes." 
        out["save_image_path"] = save_image_path
        out["img_size"] = img_size
        
        return out


if __name__ == "__main__":
    version = "trainval"
    dataroot = "./DATA/kitti"
    final_dim = (384, 1280)

    data_aug_conf = {
        "final_dim": final_dim,
        "cams": [
            "CAM_LEFT",
            "CAM_RIGHT"
        ],
        "Ncams": 1,
    }

    traindata = KittiDataset(dataroot, is_train=True,
                         data_aug_conf=data_aug_conf, max_boxes_per_sample=25)

    
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=2,
                                              shuffle=False, num_workers=16,
                                              pin_memory = (device == 'cuda'))
    
    valdata = KittiDataset(dataroot, is_train=False,
                         data_aug_conf=data_aug_conf, max_boxes_per_sample=25)

    
    valloader = torch.utils.data.DataLoader(valdata, batch_size=6,
                                              shuffle=False, num_workers=8,
                                              pin_memory = (device == 'cuda'))
    base_path = os.path.join('kitti_data')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # gen_path = os.path.join('./DATA/kitti/training/', "image_2_gt")
    gen_path = os.path.join(base_path, "testing", "image_2_gt")
    os.makedirs(gen_path, exist_ok=True)  

    max_boxes = -1


    # for batch_index, batch in enumerate(tqdm(valloader)):
    #     save_image_path = batch["save_image_path"]
    #     samples = batch["image"]
    #     image_size = batch['img_size'].cpu().numpy()
    #     b, n, c, h, w = samples.shape
        
    #     for i in range(b):
    #         for j in range(n):
    #             padded_img = TF.to_pil_image(samples[i][j] * 0.5 + 0.5)
    #             original_size = (image_size[i][j][0], image_size[i][j][1]) 
    #             padded_size = [padded_img.size[1], padded_img.size[0]]  # padded_img.size  

    #             cropped_img = crop_back_to_original(padded_img, original_size, padded_size)

    #             img_path = save_image_path[j][i]

    #             save_path = os.path.join(gen_path, img_path)
    #             par_save_path = os.path.dirname(save_path)  

    #             if not os.path.exists(par_save_path):
    #                 os.makedirs(par_save_path, exist_ok=True)  
    #             cropped_img.save(save_path)

    for batch_index, batch in enumerate(tqdm(valloader)):
        save_image_path = batch["save_image_path"]
        samples = batch["image"]
        image_size = batch['img_size'].cpu().numpy()
        b, n, c, h, w = samples.shape

        # real_images_with_box_drawing = [] # we save this durining trianing for better visualization
        # for i in range(b):
        #     images_per_cam = []
        #     for j in range(n):
        #         temp_data = {"image": batch["image"][i][j], "box":batch["box"][i][j], "box_name":batch["box_name"][j][i]}
        #         im = valdata.vis_getitem_data(out=temp_data, return_tensor=True, print_scene_description=False)
        #         images_per_cam.append(im)
        #     images_per_cam = torch.stack(images_per_cam)
        #     real_images_with_box_drawing.append(images_per_cam)
        # real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
        # real_images_with_box_drawing = rearrange(real_images_with_box_drawing, 'b n c h w -> (b n) c h w')
        
        # save_path = os.path.join(gen_path, f"batch{batch_index}.png")
        # torchvision.utils.save_image(  real_images_with_box_drawing, save_path, nrow=1, normalize=True )
        
        for i in range(b):
            for j in range(n):
               
                original_size = (image_size[i][j][0], image_size[i][j][1]) 
                padded_size = [h, w]  # padded_img.size  

                img_path = save_image_path[j][i]

                save_path = os.path.join(gen_path, img_path)
                par_save_path = os.path.dirname(save_path)  

                if not os.path.exists(par_save_path):
                    os.makedirs(par_save_path, exist_ok=True)  
                
                temp_data = {"image": batch["image"][i][j], "box":batch["box"][i][j], "box_name":batch["box_name"][j][i]}
                padded_img_with_box_drawing = valdata.vis_getitem_data(out=temp_data, return_tensor=True, print_scene_description=False)
                padded_img_with_box_drawing = TF.to_pil_image(padded_img_with_box_drawing)
                cropped_img = crop_back_to_original(padded_img_with_box_drawing, original_size, padded_size)
                cropped_img.save(save_path)

