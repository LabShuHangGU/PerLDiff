# Standard library imports
import os
import json
import math
import time
import shutil
import random
import argparse
from datetime import datetime
from collections import defaultdict, OrderedDict
from copy import deepcopy
from functools import partial

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TVF
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw
import gradio as gr
import cv2
from einops import rearrange, repeat
from transformers import (
    CLIPProcessor,
    CLIPModel,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from pyquaternion import Quaternion

# For distributed training (if NVIDIA's apex is available)
try:
    from apex import amp
except ImportError:
    amp = None

# Local application/library specific imports
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.attention import GatedCrossAttentionDense
from ldm.util import instantiate_from_config, count_params
from dataset.concat_dataset import ConCatDataset
from dataset.utils import (
    yaw_to_quaternion,
    quaternion_to_yaw,
    color_to_rgb,
    draw_box_3d,
)
from scripts.distributed import get_rank, synchronize, get_world_size
from scripts.convert_ckpt import add_additional_channels
from scripts.gradio_util import check_class_name, vis_getitem_data
from trainer import (
    ImageCaptionSaver,
    read_official_convnext_ckpt,
    read_official_gligen_ckpt,
    read_official_sd_ckpt,
    batch_to_device,
    sub_batch,
    disable_grads,
    update_ema,
    create_expt_folder_with_auto_resuming,
)

from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
from valer import read_val_ckpt

# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #
def set_alpha_scale(model, alpha_scale):
    
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense:
            module.scale = alpha_scale

def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas


def read_val_ckpt(ckpt_path):
    "Read val ckpt and convert into my style"
    print("\n" + "*" * 20 + " load model from {}!".format(ckpt_path) + " *" * 20 + "\n")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    return state_dict


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 


class Visualizer(object):
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        # config.model.params.writer_name = self.name
        self.model = instantiate_from_config(config.model).to(self.device)

        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)


        self.convnext = instantiate_from_config(config.convnext).to(self.device)
        convnext_tiny_checkpoint = read_official_convnext_ckpt(os.path.join(config.DATA_ROOT, 'convnext_tiny_1k_224_ema.pth') )
        convnext_tiny_checkpoint['model'].pop('head.weight')
        convnext_tiny_checkpoint['model'].pop('head.bias')
        self.convnext.load_state_dict( convnext_tiny_checkpoint['model'] )
        
        state_dict = read_val_ckpt( config.val_ckpt_name  )

        if self.config.official_ckpt_name == "sd-v1-4.ckpt":
            official_state_dict = read_official_sd_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
        else:
            official_state_dict = read_official_gligen_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
    

        # load original GLIGEN ckpt (with inuput conv may be modified) 
        missing_keys, unexpected_keys = self.model.load_state_dict( state_dict["model"], strict=False  )
        assert unexpected_keys == []
        original_params_names = list( state_dict["model"].keys()  ) # used for sanity check later 
        
               
        self.autoencoder.load_state_dict( official_state_dict["autoencoder"]  )
        self.text_encoder.load_state_dict( official_state_dict["text_encoder"]  )
        self.diffusion.load_state_dict( official_state_dict["diffusion"]  )
 
        self.autoencoder.eval()
        self.text_encoder.eval()
        self.convnext.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)
        disable_grads(self.convnext)

        if get_rank() == 0:
            count_params(self.autoencoder, verbose=True)
            count_params(self.text_encoder, verbose=True)
            count_params(self.diffusion, verbose=True)
            count_params(self.model, verbose=True)
        
        self.pil_to_tensor = transforms.PILToTensor()

        self.max_boxes = 80

        self.cam_types = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
   
        self.class_names = ['car', 'truck', 'bus', 'bicycle', 'motorcycle']
        self.controlling_condition_input = instantiate_from_config(config.controlling_condition_input)
        self.model.controlling_condition_input = self.controlling_condition_input
        
    @torch.no_grad()
    def apdate_batch(self, batch):
        box_name = np.array(batch['box_name'])
        b, num_cam, num_box= box_name.shape
        box_name = list(box_name.reshape(-1))

        _, text_feature = self.text_encoder.encode(box_name, return_pooler_output=True)
        text_feature = rearrange(text_feature, '(b n m) c -> b n m c', n=num_cam, b=b, m=num_box)
        batch['box_text_embedding'] = text_feature

        B, N, C, H, W = batch["road_map"].shape

        road_map = rearrange(batch["road_map"], 'b n c h w -> (b n) c h w')
        uroad_map = torch.ones_like(road_map, dtype=road_map.dtype).to(road_map.device)

        road_map_embedding = self.convnext(road_map)
        road_map_embedding = rearrange(road_map_embedding, '(b n) c -> b n c', b=B,n=N)

        uroad_map_embedding = self.convnext(uroad_map)
        uroad_map_embedding = rearrange(uroad_map_embedding, '(b n) c -> b n c', b=B,n=N)

        # del batch['road_map']
        # del batch['box_name']

        context = self.text_encoder.encode( batch["scene_description"]  )
        batch['context'] = context

        batch['ucontext'] = self.text_encoder.encode( context.shape[0] * [""] )

        del batch['scene_description']

        batch["road_map_embedding"] = road_map_embedding
        batch["uroad_map_embedding"] = uroad_map_embedding
        
        return batch


    def add_batch_size(self, batch, num_samples=1):

        batch["box_mask"] = batch["box_mask"][None, :, :].repeat(num_samples, 1, 1)
        batch['box_name'] = [batch['box_name']]
        batch["box"] = batch["box"][None, :, :, :].repeat(num_samples, 1, 1, 1)
        batch["perl_box_masking_map"] = batch["perl_box_masking_map"][None, :, :, :, :].repeat(num_samples, 1, 1, 1, 1)
        batch["road_map"] = batch["road_map"][None, :, :, :, :].repeat(num_samples, 1, 1, 1, 1)
        batch["perl_road_masking_map"] = batch["perl_road_masking_map"][None, :, :, :].repeat(num_samples, 1, 1, 1)
        batch["scene_description"] = [batch["scene_description"]]
        return batch


    def start_vis(self, json_file_path=None, rot=0.0, pos=0.0, size=0.0, scene_description=None, guidance_scale_c=5.0, step=50):
        print(f"start_vis:, rot:{rot}, pos:{pos}, size:{size}")

        self.model.eval()
        batch = self.batch = self.prepare_batch(json_file_path=json_file_path, scene_description=scene_description, rot=rot, pos=pos, size=size)

        batch_to_device(batch, self.device)
        with torch.no_grad():

            batch = self.add_batch_size(batch)
            batch = self.apdate_batch(batch)
                    
            # model_wo_wrapper = self.model.module if self.config.distributed else self.model
            model_wo_wrapper = self.model

            if self.config.plms:
                sampler = PLMSSampler(self.diffusion, model_wo_wrapper)
            else:
                sampler = DDIMSampler(self.diffusion, model_wo_wrapper)

            B, N, C, H, W = batch['road_map'].shape    

            shape = (B, model_wo_wrapper.num_camera, model_wo_wrapper.in_channels, model_wo_wrapper.image_size[0], model_wo_wrapper.image_size[1])
                
            controlling_condition_input = self.controlling_condition_input.prepare(batch)
            input = dict( x=None, 
                        timesteps=None, 
                        controlling_condition_input=controlling_condition_input)
                
            # z0 = None # used for replacing known region in diffusion process
            # alpha_generator_func = partial(alpha_generator, type=[0.8,0.1,0.1])
            # alpha_generator_func = partial(alpha_generator, type=[1,0,0])

                
            samples = sampler.sample(S=step, shape=shape, input=input, guidance_scale_c=guidance_scale_c, mask=None, x0=None)
                
            autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
            samples = autoencoder_wo_wrapper.decode(samples).cpu()
            samples = torch.clamp(samples, min=-1, max=1)

            # we save this durining trianing for better visualization
            sample_images_with_box_drawing = []

            for i in range(B):
                images_per_cam = []
                for j in range(N):
                    temp_data = {"image": samples[i][j], "box":batch["box"][i][j]}
                    
                    im = vis_getitem_data(
                        out=temp_data, return_tensor=True, print_scene_description=False)
                    # images_per_cam.append(samples[i][j]*0.5 + 0.5)
                    images_per_cam.append(im)
                images_per_cam = torch.stack(images_per_cam)
                sample_images_with_box_drawing.append(images_per_cam)
            sample_images_with_box_drawing = torch.stack(
                sample_images_with_box_drawing)
            
            return sample_images_with_box_drawing[0]


    def draw_image(self, batch=None):
        road_maps = []
        image_gts = []

        N, C, H, W = batch['road_map'].shape

        for i in range(N):
            temp_data = {"image": batch['road_map'][i], "box":batch["box"][i]}
            road_map = vis_getitem_data(
                    out=temp_data, return_tensor=True, print_scene_description=False)
            temp_data = {"image": batch['image'][i], "box":batch["box"][i]}
            image_gt = vis_getitem_data(
                    out=temp_data, return_tensor=True, print_scene_description=False)
            road_maps.append(road_map)
            image_gts.append(image_gt)
        road_maps = torch.stack(road_maps)
        image_gts = torch.stack(image_gts)

        return road_maps, image_gts 

    def prepare_batch_info(self, json_info=None, rot=0.0, pos=0.0, size=0.0):
        print(f"prepare_batch_info:, rot:{rot}, pos:{pos}, size:{size}")
        
        final_dim = [256, 384]
        # Crop 900 * 1600 images to 900 * 1600
        img_size = [900, 1600]
        crop_size = [img_size[0], img_size[1]]
        sample_image = []
        sample_road_map = []
        sample_box_name = []
        sample_box_mask = []
        sample_box_corner_3d = []
        
        sample_perl_box_masking_map = []
        sample_perl_road_masking_map = []


        for id_cam, cam in enumerate(self.cam_types):
            box_name = ["" for _ in range(self.max_boxes)]
            box_mask = torch.zeros(self.max_boxes)
            box_corner_3d = torch.zeros(self.max_boxes, 16)

            perl_box_masking_maps = torch.zeros(self.max_boxes, final_dim[0], final_dim[1])

            road_map_name = os.path.join("DATA",'nuscenes', json_info[cam]['road_map_name'])
            road_map = Image.open(road_map_name).convert("RGB")

            image = Image.open(road_map_name.replace("samples_road_map", "samples")).convert("RGB")

            camera_intrinsic = np.array(json_info[cam]["intrinsic"])

            annotation_index = 0

            for i, box_info in enumerate(json_info[cam]['anns']):
                name = box_info['box_name']

                flag = check_class_name(name, self.class_names)
                if size != 0.0 and flag:
                    box_size = [x * size for x in box_info['box_size'][:]]
                else:
                    box_size = box_info['box_size']

                if pos != 0.0 and flag:
                    box_center = [x + pos for x in box_info['box_center'][:1]] + box_info['box_center'][1:]
                else:
                    box_center = box_info['box_center']

                if rot != 0.0 and flag:
                    box_degree = (box_info['box_degree'] + rot) % 360
                    if box_degree > 180:
                        box_degree -= 360
                    elif box_degree <= -180:
                        box_degree += 360
                else:
                    box_degree = box_info['box_degree']

                box_rotation = Quaternion(yaw_to_quaternion(box_degree))
                box = Box(center=box_center, size=box_size, orientation=box_rotation, name=name)

                box.translate(-np.array(json_info[cam]
                                                ["ego2sensor_translation"]))
                box.rotate(Quaternion(
                            json_info[cam]["ego2sensor_rotation"]).inverse)

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
                    
                bbox_3d = corner_coords[:2, :].copy()
                    
                dx = (img_size[1] / 2 - crop_size[1] / 2)
                dy = (img_size[0] / 2 - crop_size[0] / 2)

                # because crop, we must define new coordinate of x,y
                new_min_x = min_x - dx
                new_max_x = max_x - dx

                new_min_y = min_y - dy
                new_max_y = max_y - dy

                bbox_3d[0, :] = (bbox_3d[0, :] - dx)  / crop_size[1]
                bbox_3d[1, :] = (bbox_3d[1, :] - dy) / crop_size[0]

                if new_max_x <= 0 or new_max_y <= 0 or new_min_x >= crop_size[1] or new_min_y >= crop_size[0]:
                    continue
                    
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
                
                box_name[annotation_index] = name
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
        )

    def prepare_batch(self, json_file_path=None, scene_description=None, rot=0.0, pos=0.0, size=0.0):
        print(f"prepare_batch:, rot:{rot}, pos:{pos}, size:{size}")
        with open(os.path.join( json_file_path), 'r', encoding='UTF-8') as f:
            json_info = json.load(f)

        out = {}
        (   image,
            road_map,
            box_name,
            box_mask,
            box_corner_3d,
            perl_box_masking_map,
            perl_road_masking_map,
        ) = self.prepare_batch_info(json_info=json_info, rot=rot, pos=pos, size=size)

        out["image"] = image
        out["road_map"] = road_map
        out["box_name"] = box_name
        out["box_mask"] = box_mask
        out['box'] = box_corner_3d

        out["perl_box_masking_map"] = perl_box_masking_map
        out["perl_road_masking_map"] = perl_road_masking_map

        if scene_description != None:
            out["scene_description"] = scene_description

        return out

    def prepare_road_map_with_draw(self, json_file_path=None, rot=0.0, pos=0.0, size=0.0):
        print(f"prepare_road_map_with_draw:, rot:{rot}, pos:{pos}, size:{size}")
        batch = self.prepare_batch(json_file_path=json_file_path, rot=rot, pos=pos, size=size)
        return self.draw_image(batch)
