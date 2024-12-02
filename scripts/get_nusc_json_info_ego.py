# Standard library imports
import os
import sys
import math
import random
import json
from glob import glob
from collections import defaultdict, OrderedDict

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset, DistributedSampler
from einops import rearrange, repeat
from transformers import CLIPProcessor, CLIPModel
import cv2
from pyquaternion import Quaternion

# NuScenes specific imports
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords

# Local application/library specific imports
sys.path.append('.')
from dataset.utils import yaw_to_quaternion, quaternion_to_yaw, color_to_rgb, draw_box_3d


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
            # "v1.0-trainval": {True: "train", False: "val"},Actually, there are also test sets
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
        # samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        return samples

    def get_camera_info(self, sample_record, cams, index=None):

        scene_description = self.nusc.get('scene', sample_record['scene_token'])[
            'description']

        nusc_sample_info = {}
       
        for cam in cams:

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

            ##################################################################
            nusc_sample_info[cam] = {}
            # nusc_sample_info[cam]["global2ego_translation"] = pose_record["translation"]
            # nusc_sample_info[cam]["global2ego_rotation"] = pose_record["rotation"]
            nusc_sample_info[cam]["intrinsic"] = cs_record["camera_intrinsic"]
            nusc_sample_info[cam]["ego2sensor_translation"] = cs_record["translation"]
            nusc_sample_info[cam]["ego2sensor_rotation"] = cs_record["rotation"]
            ####################################################################

            nusc_sample_info[cam]['road_map_name'] = sd_record["filename"].replace("samples", "samples_road_map")
            
            # get all the annotation
            ann_records = [
                self.nusc.get("sample_annotation", token) for token in s_record["anns"]
            ]
            track_class = ["car", "bus", "truck", "trailer", "motorcycle", "bicycle", "construction", "pedestrian", "barrier", "trafficcone"]
            
            nusc_sample_info[cam]['anns'] = []

            
            # annotation_index = 0
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
                    instance_name = "construction vehicle"
                elif instance_name == "trafficcone":
                    instance_name = "traffic cone"

                box = self.nusc.get_box(ann_record["token"])

                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record["rotation"]).inverse)

                box_name = instance_name
                box_size = list(box.wlh)
                box_center = list(box.center)

                box_degree = quaternion_to_yaw(box.orientation)

                print(f"box_name = {box_name}")
                print(f"box_center = {box_center}")
                print(f"box_degree = {box_degree}")
                print(f"box_size = {box_size}")

                box_info = {}
                box_info['box_name'] = box_name
                box_info['box_center'] = box_center
                box_info['box_degree'] = box_degree
                box_info['box_size'] = box_size
                nusc_sample_info[cam]['anns'].append(box_info)
            
        return nusc_sample_info
        
        
 
    def choose_cams(self):
        if self.data_aug_conf["Ncams"] < len(
            self.data_aug_conf["cams"]
        ):
            cams = [self.data_aug_conf["cams"][1]]
        else:
            cams = self.data_aug_conf["cams"]
        return cams
    
    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_data_record = self.samples[index]
        
        cams = self.choose_cams()

        nusc_sample_info = self.get_camera_info(sample_data_record, cams)
        # #########################################
        # cams_params_json = f"samples_{index}_info.json"
        # with open(cams_params_json, "w") as f:
        #     json.dump(nusc_sample_info, f, indent=4)
        # #########################################

        return nusc_sample_info
    

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
    # traindata = NuscDataset(version, dataroot, is_train=True,
    #                      data_aug_conf=data_aug_conf, max_boxes_per_sample=80)
    

    valdata = NuscDataset(version, dataroot, is_train=False,
                         data_aug_conf=data_aug_conf, max_boxes_per_sample=80)

    save_dir = "nuscenes_sample_info_files"
    os.makedirs(save_dir, exist_ok=True)
    
    for i, j in enumerate(range(500, 520)):
        nusc_sample_info = valdata[j]
        #########################################
        cams_params_json = os.path.join(save_dir, f"samples_{i}_info.json")
        with open(cams_params_json, "w") as f:
            json.dump(nusc_sample_info, f, indent=4)
        #########################################

    
    


