import os
import math
import json
import random
import shutil
import time
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from einops import rearrange, repeat

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.attention import GatedCrossAttentionDense
from ldm.util import instantiate_from_config

from dataset.concat_dataset import ConCatDataset  # , collate_fn
from scripts.distributed import get_rank, synchronize, get_world_size
from trainer import (
    ImageCaptionSaver,
    read_official_convnext_ckpt,
    read_official_gligen_ckpt,
    read_official_sd_ckpt,
    batch_to_device,
    sub_batch,
    disable_grads,
    count_params,
    update_ema,
    create_expt_folder_with_auto_resuming,
)

try:
    from apex import amp
except ImportError:
    pass

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


class Gener:
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            self.config_dict = vars(config)
            torch.save(  self.config_dict,  os.path.join(self.name, "config_dict.pth")     )

        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
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
        # original_params_names = list( state_dict["model"].keys()  ) # used for sanity check later 
        
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


        # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #  
        dataset_val = ConCatDataset(config.val_dataset_names, config.DATA_ROOT, train=False, repeats=None)
        dataset_train = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, train=True, repeats=None)
        ##########################
        num_val_samples = len(dataset_val)
        num_train_samples = len(dataset_train)
        
        sub_val_indices = list(range(num_val_samples-config.total_batch_size, num_val_samples))
        sub_train_indices = list(range(num_train_samples-config.total_batch_size, num_train_samples))

        
        sub_dataset_val = Subset(dataset_val, sub_val_indices)
        sub_sampler_val = DistributedSampler(sub_dataset_val, seed=config.seed, shuffle=False) if config.distributed else None 
        sub_loader_val = DataLoader(sub_dataset_val,  batch_size=config.batch_size, 
                                                   shuffle=False,
                                                   num_workers=config.workers, 
                                                   pin_memory=False, 
                                                   sampler=sub_sampler_val,
                                                   drop_last=True)
        
        sub_dataset_train = Subset(dataset_train, sub_train_indices)
        sub_sampler_train = DistributedSampler(sub_dataset_train, seed=config.seed, shuffle=False) if config.distributed else None 
        sub_loader_train = DataLoader(sub_dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=False,
                                                   num_workers=config.workers, 
                                                   pin_memory=False, 
                                                   sampler=sub_sampler_train,
                                                   drop_last=True)
        
        sampler_val = DistributedSampler(dataset_val, seed=config.seed, shuffle=False) if config.distributed else None 
        loader_val = DataLoader(dataset_val,  batch_size=config.batch_size, 
                                                   shuffle=False,
                                                   num_workers=config.workers, 
                                                   pin_memory=False, 
                                                   sampler=sampler_val,
                                                   drop_last=True)
        sampler_train = DistributedSampler(dataset_train, seed=config.seed, shuffle=False) if config.distributed else None 
        loader_train = DataLoader(dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=False,
                                                   num_workers=config.workers, 
                                                   pin_memory=False, 
                                                   sampler=sampler_train,
                                                   drop_last=True)
        self.dataset_val = dataset_val
        self.loader_val = loader_val

        self.dataset_train = dataset_train
        self.loader_train = loader_train

        self.sub_dataset_val = sub_dataset_val
        self.sub_loader_val = sub_loader_val

        self.sub_dataset_train = sub_dataset_train
        self.sub_loader_train = sub_loader_train

        if get_rank() == 0:
            total_val_image = dataset_val.total_images()
            print("Total validation images: ", total_val_image)     
            total_train_iamge = dataset_train.total_images()
            print("Total training images: ", total_train_iamge)
        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#    
        
        # func return input for controlling condition encoding
        self.controlling_condition_input = instantiate_from_config(config.controlling_condition_input)
        self.model.controlling_condition_input = self.controlling_condition_input
 
        dirs = os.path.join(self.name, "validation")
        if not os.path.exists(dirs):
            os.makedirs(dirs, exist_ok=True)
       
        self.image_caption_saver = ImageCaptionSaver(base_path=dirs, nrow=6)
        
        if config.distributed:
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False )


    @torch.no_grad()
    def apdate_batch(self, batch):
        box_name = np.array(batch['box_name'])
        num_cam, num_box, b = box_name.shape
        box_name = list(box_name.reshape(-1))

        _, text_feature = self.text_encoder.encode(box_name, return_pooler_output=True)
        text_feature = rearrange(text_feature, '(n m b) c -> n m b c', n=num_cam, b=b, m=num_box)
        batch['box_text_embedding'] = rearrange(text_feature, 'n m b c -> b n m c')

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
    
    @torch.no_grad()
    def get_input(self, batch):
        z = self.autoencoder.encode( batch["image"] )

        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999

        return z, t



    def start_validation(self, is_train=False):

        self.model.eval()
        if not is_train:
            data = [self.loader_val, self.sub_loader_val]
        else:
            data = [self.loader_train, self.sub_loader_train]
        
        for loader in data:
            for iter_idx, batch in enumerate(tqdm(loader)):
                self.iter_idx = iter_idx

                batch_to_device(batch, self.device)
                with torch.no_grad():
                    batch = self.apdate_batch(batch)
                    
                model_wo_wrapper = self.model.module if self.config.distributed else self.model

                # iter_name = self.iter_idx + 1     # we add 1 as the actual name
                
                # Do inference 
                batch_here = self.config.batch_size

                if self.config.plms:
                    sampler = PLMSSampler(self.diffusion, model_wo_wrapper)
                else:
                    sampler = DDIMSampler(self.diffusion, model_wo_wrapper)
                    

                shape = (batch_here, model_wo_wrapper.num_camera, model_wo_wrapper.in_channels, model_wo_wrapper.image_size[0], model_wo_wrapper.image_size[1])
                
                controlling_condition_input = self.controlling_condition_input.prepare(batch)
                input = dict(x=None, 
                            timesteps=None, 
                            controlling_condition_input=controlling_condition_input)
                
                z0 = None # used for replacing known region in diffusion process
                # alpha_generator_func = partial(alpha_generator, type=[0.8,0.1,0.1])
                # alpha_generator_func = partial(alpha_generator, type=[1,0,0])

                
                samples = sampler.sample(S=self.config.step, shape=shape, input=input, guidance_scale_c=self.config.guidance_scale_c, mask=None, x0=z0)
                
                autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
                samples = autoencoder_wo_wrapper.decode(samples).cpu()
                samples = torch.clamp(samples, min=-1, max=1)

                save_image_path = batch["save_image_path"]
                b, n, c, h, w = samples.shape

                assert batch_here == b

                cameras =  ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']


                for i in range(b):
                    for j in range(n):
                        img = TF.to_pil_image(samples[i][j] * 0.5 + 0.5)
                        
                        if n > 1:
                            cam = cameras[j]
                            img_path = save_image_path[cam][i]
                        else:
                            img_path = save_image_path[j][i]

                        save_path = os.path.join(self.config.gen_path, img_path.replace('jpg', "png"))
                        par_save_path = os.path.dirname(save_path)  

                        if not os.path.exists(par_save_path):
                            os.makedirs(par_save_path, exist_ok=True)  
                        img.save(save_path)
                
                if iter_idx % 100 == 0 and get_rank() == 0:
                    print(f"**************** Validation {iter_idx} / {len(loader)} *****************\n")
                synchronize()
            synchronize()
        synchronize()
        print("Validation finished. Start exiting")
        exit()

