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

from dataset.utils import yaw_to_quaternion, quaternion_to_yaw, color_to_rgb, draw_box_3d, get_color, quaternion_multiply

device = torch.device("cuda")


class NuscDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        version,
        dataroot,
        is_train,
        data_aug_conf,
        max_boxes_per_sample=0,
        random_crop = False,
        random_flip = True,
    ):
        self.nusc = NuScenes(
            version="v1.0-{}".format(version),
            # dataroot=os.path.join(dataroot, version),
            dataroot=os.path.join(dataroot),
            verbose=False,
        )

        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.random_flip = random_flip
        self.pil_to_tensor = transforms.PILToTensor()
        self.max_boxes = max_boxes_per_sample
        self.scenes = self.get_scenes()
        self.samples = self.prepro()

    def total_images(self):
        return len(self)

    def get_scenes(self):
        # filter by scene split
        split = {
            "v1.0-trainval": {True: "train", False: "val"},
            "v1.0-mini": {True: "mini_train", False: "mini_val"},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]
        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [
            samp
            for samp in samples
            if self.nusc.get("scene", samp["scene_token"])["name"] in self.scenes
        ]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        return samples

    def get_camera_info(self, sample_record, cams):

        save_image_path = OrderedDict()
        final_dim = [self.data_aug_conf["final_dim"][0], self.data_aug_conf["final_dim"][1]]

        sample_image = []
        sample_road_map = []
        sample_box_name = []
        sample_box_mask = []
        sample_box_corner_3d = []
        
        sample_perl_box_masking_map = []
        sample_perl_road_masking_map = []

        scene_description = self.nusc.get('scene', sample_record['scene_token'])[
            'description']

        # cams_params = {}
        
        for cam in cams:
            box_name = ["" for _ in range(self.max_boxes)]
            box_mask = torch.zeros(self.max_boxes)
            box_corner_3d = torch.zeros(self.max_boxes, 16)

            perl_box_masking_maps = torch.zeros(self.max_boxes, final_dim[0], final_dim[1])

            sample_data_token = sample_record["data"][cam]
            # Retrieve sensor & pose records
            sd_record = self.nusc.get("sample_data", sample_data_token)

            if not sd_record["is_key_frame"]:
                raise ValueError("not keyframes")

            s_record = self.nusc.get("sample", sd_record["sample_token"])
            cs_record = self.nusc.get(
                "calibrated_sensor", sd_record["calibrated_sensor_token"]
            )
            pose_record = self.nusc.get(
                "ego_pose", sd_record["ego_pose_token"])
            camera_intrinsic = np.array(cs_record["camera_intrinsic"])

            # ##################################################################
            # cams_params[cam] = {}
            # cams_params[cam]["global2ego_translation"] = pose_record["translation"]
            # cams_params[cam]["global2ego_rotation"] = pose_record["rotation"]
            # cams_params[cam]["intrinsic"] = cs_record["camera_intrinsic"]
            # cams_params[cam]["ego2sensor_translation"] = cs_record["translation"]
            # cams_params[cam]["ego2sensor_rotation"] = cs_record["rotation"]
            # ####################################################################

            img_size = (sd_record["height"], sd_record["width"])
            img_name = os.path.join(self.nusc.dataroot, sd_record["filename"])
            image = Image.open(img_name).convert("RGB")

            save_image_path[cam] = sd_record["filename"]

            road_map_name = img_name.replace("samples", "samples_road_map")
            road_map = Image.open(road_map_name).convert("RGB")

            # for 256x384 followed by BEVFormer, we resize the image to 256x384, not crop
            crop_size = [img_size[0], img_size[1]]
            
            # get all the annotation
            ann_records = [
                self.nusc.get("sample_annotation", token) for token in s_record["anns"]
            ]
            
            track_class = ["car", "bus", "truck", "trailer", "motorcycle", "bicycle", "construction", "pedestrian", "barrier", "trafficcone"]
            
            
            annotation_index = 0
            for ann_record in ann_records:
                # get the boxes
                ann_record["sample_annotation_token"] = ann_record["token"]
                ann_record["sample_data_token"] = sample_data_token

                instance_name = None
                instance_names = ann_record["category_name"].split(".")
                for name in instance_names:
                    if name in track_class:
                        instance_name = name
                        continue
                if instance_name == None:
                    continue
                
                if instance_name == "construction":
                    instance_name = "construction_vehicle"
                elif instance_name == "trafficcone":
                    instance_name = "traffic_cone"
                    
                # json_box_name = json.dumps(ann_record["category_name"], indent=4)
                # with open( "camera_bbox_info.json" ,"a") as json_file:
                #     json_file.write(ann_record["category_name"] + '\n')

                box = self.nusc.get_box(ann_record["token"])
                
                # Move box to ego vehicle coord system. 
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record["translation"]))
                box.rotate(Quaternion(cs_record["rotation"]).inverse)
                # Filter out the corners that are not in front of the calibrated sensor.
                corners_3d = box.corners()
                in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                corners_3d = corners_3d[:, in_front]
                if corners_3d.shape[1] != 8:
                    continue


                # Project 3d box to 2d.
                corner_coords = view_points(corners_3d, camera_intrinsic, True)[:2,:]
                corner_coords_2d = post_process_coords(corner_coords.T.tolist())
                if corner_coords_2d is None:
                    continue
                else:  
                    min_x, min_y, max_x, max_y = corner_coords_2d

                dx = (img_size[1]  - crop_size[1]) / 2
                dy = (img_size[0] - crop_size[0]) / 2

                bbox_3d = corner_coords[:2, :].copy()

                # because crop, we must define new coordinate of x,y
                new_min_x = min_x - dx
                new_max_x = max_x - dx

                new_min_y = min_y - dy
                new_max_y = max_y - dy

                if new_max_x <= 0 or new_max_y <= 0 or new_min_x >= crop_size[1] or new_min_y >= crop_size[0]:
                    continue

                bbox_3d[0, :] = (bbox_3d[0, :] - dx)  / crop_size[1]
                bbox_3d[1, :] = (bbox_3d[1, :] - dy) / crop_size[0]    

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

            left = img_size[1] // 2 - crop_size[1] // 2
            right = img_size[1] // 2 + crop_size[1] // 2

            upper = img_size[0] // 2 - crop_size[0] // 2
            lower = img_size[0] // 2 + crop_size[0] // 2

            image = image.crop((left, upper, right, lower))
            image = image.resize((final_dim[1], final_dim[0]))

            road_map = road_map.crop((left, upper, right, lower))
            road_map = road_map.resize((final_dim[1], final_dim[0]))

            road_map_np = np.array(road_map)

            low_white = np.array([200, 200, 200], dtype=np.uint8)
            up_white = np.array([255, 255, 255], dtype=np.uint8)
            road_map_mask = cv2.inRange(road_map_np, low_white, up_white)

            perl_road_masking_map = cv2.bitwise_not(road_map_mask)

            image = (self.pil_to_tensor(image).float() / 255 - 0.5) / 0.5
            road_map = (self.pil_to_tensor(road_map).float() / 255 - 0.5) / 0.5

            perl_road_masking_map = (torch.tensor(perl_road_masking_map, dtype=image.dtype)) / 255
            
            sample_image.append(image)
            sample_road_map.append(road_map)
            
            sample_perl_box_masking_map.append(perl_box_masking_maps)

            sample_box_mask.append(box_mask)
            sample_box_name.append(box_name)
            sample_box_corner_3d.append(box_corner_3d)
            sample_perl_road_masking_map.append(perl_road_masking_map)

        sample_image = torch.stack(sample_image)
        sample_road_map = torch.stack(sample_road_map)

        sample_perl_box_masking_map = torch.stack(sample_perl_box_masking_map)
        sample_box_mask = torch.stack(sample_box_mask)
        sample_box_corner_3d = torch.stack(sample_box_corner_3d)
        sample_perl_road_masking_map = torch.stack(sample_perl_road_masking_map)


        return (
            sample_image,
            sample_road_map,
            sample_box_name,
            sample_box_mask,
            sample_box_corner_3d,
            sample_perl_box_masking_map,
            sample_perl_road_masking_map,
            scene_description,
            save_image_path,
        )

    def choose_cams(self):
        if self.data_aug_conf["Ncams"] < len(
            self.data_aug_conf["cams"]
        ):
            cams = [self.data_aug_conf["cams"][1]]
        else:
            cams = self.data_aug_conf["cams"]
        return cams
    
    def vis_getitem_data(self, index=None, out=None, return_tensor=False, name="res.jpg", print_scene_description=False):

        img = torchvision.transforms.functional.to_pil_image( out["image"]*0.5 + 0.5 )
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

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_data_record = self.samples[index]

        cams = self.choose_cams()

        out = {}
        (   image,
            road_map,
            box_name,
            box_mask,
            box_corner_3d,
            perl_box_masking_map,
            perl_road_masking_map,
            scene_description,
            save_image_path, 
        ) = self.get_camera_info(sample_data_record, cams)
        
        out["image"] = image
        out["road_map"] = road_map
        out["box_name"] = box_name
        out["box_mask"] = box_mask
        out['box'] = box_corner_3d

        out["perl_box_masking_map"] = perl_box_masking_map
        out["perl_road_masking_map"] = perl_road_masking_map

        lower_scene_description = scene_description.lower()

        has_night = "night" in lower_scene_description
        has_rain = "rain" in lower_scene_description

        if has_night and has_rain:
            out["scene_description"] = "Realistic autonomous driving scene, " + "night and rain."
        elif has_night:
            out["scene_description"] = "Realistic autonomous driving scene, " + "night."
        elif has_rain:
            out["scene_description"] = "Realistic autonomous driving scene, " + "rain."
        else:
            out["scene_description"] = "Realistic autonomous driving scene, " + "day."

        out["save_image_path"] = save_image_path
        return out


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
    valloader = torch.utils.data.DataLoader(valdata, batch_size=1,
                                              shuffle=False, num_workers=8,
                                              pin_memory = False)

    
    base_path = os.path.join('test', 'vis_sample_256x384')
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    for batch_index, batch in enumerate(valloader):

        print(batch['scene_description'])
        road_map = batch["road_map"]
        print(f"road map : {road_map.shape, road_map.min(), road_map.max()}")
        road_map = rearrange(road_map, "b n c h w->(b n) c h w")
        torchvision.utils.save_image(
                    road_map * 0.5 + 0.5, os.path.join(base_path, "road_map_{}.png".format(batch_index)), nrow=6)
        
        
        image = batch["image"]
        batch_here, num_cam, _, _, _ = image.shape
        real_sample_image_with_box_drawing = [] # we save this durining trianing for better visualization
        for i in range(batch_here):
            sample_image_per_cam = []
            for j in range(num_cam):
                temp_data = {"image": batch["image"][i][j], "box":batch["box"][i][j], "box_name":batch["box_name"][i][j]}
                im = valdata.vis_getitem_data(out=temp_data, return_tensor=True, print_scene_description=False)
                sample_image_per_cam.append(im)
            sample_image_per_cam = torch.stack(sample_image_per_cam)
            real_sample_image_with_box_drawing.append(sample_image_per_cam)
        real_sample_image_with_box_drawing = torch.stack(real_sample_image_with_box_drawing)
        print(f"image {image.shape, image.min(), image.max()}")
        real_sample_image_with_box_drawing = rearrange(real_sample_image_with_box_drawing, "b n c h w->(b n) c h w")
        torchvision.utils.save_image(
                    real_sample_image_with_box_drawing , os.path.join(base_path, "image_{}.png".format(batch_index)), nrow=6)
        
        perl_box_masking_map = batch["perl_box_masking_map"]
        print(f"perl_box_masking_map {perl_box_masking_map.shape, perl_box_masking_map.min(), perl_box_masking_map.max()}")
        perl_box_masking_map = rearrange(perl_box_masking_map[:, :, :, None, :, :], "b n m c h w->(b n m) c h w")
        torchvision.utils.save_image(
                    perl_box_masking_map , os.path.join(base_path, "perl_box_masking_map_{}.png".format(batch_index)), nrow=80)

        perl_road_masking_map = batch["perl_road_masking_map"]
        print(f"perl_road_masking_map {perl_road_masking_map.shape, perl_road_masking_map.min(), perl_road_masking_map.max()}")
        perl_road_masking_map = rearrange(perl_road_masking_map[:, :, None, :, :], "b n c h w->(b n) c h w")
        torchvision.utils.save_image(
                    perl_road_masking_map , os.path.join(base_path, "perl_road_masking_map_{}.png".format(batch_index)), nrow=6)
    
        if batch_index > 2:
            break
    
    


