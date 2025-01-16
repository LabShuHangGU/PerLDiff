import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torch.nn import functional as F
from scripts.distributed import get_rank, synchronize, get_world_size
from torch.utils.tensorboard import SummaryWriter
import os
from einops import rearrange, repeat
import torchvision
from torchvision import datasets, transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from gaussian_smothing import GaussianSmoothing
from ldm import util          


class DDIMSampler(object):
    def __init__(self, diffusion, model, schedule="linear", alpha_generator_func=None, set_alpha_scale=None, name=None):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.device = diffusion.betas.device
        self.ddpm_num_timesteps = diffusion.num_timesteps
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func  # func
        self.set_alpha_scale = set_alpha_scale  # func

        self.name = name

        self.iter_idx = 0

        # Using the boxdiffusion method
        util._init_()

        # if get_rank() == 0:
        if not os.path.exists(os.path.join(self.name, 'Log')):
            os.makedirs(os.path.join(self.name, 'Log'))
        
        self.writer = SummaryWriter(os.path.join(self.name, 'Log'))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=False)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        def to_torch(x): return x.clone().detach().to(
            torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(
            self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(
            np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(
            np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=False)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self, S, shape, input, uc=None, uimage_back=None, guidance_scale=1, mask=None, x0=None,
               attention_res=[32, 48]
               ):
        self.make_schedule(ddim_num_steps=S)
        return self.ddim_sampling(shape, input, uc,uimage_back, guidance_scale,  mask=mask, x0=x0,
                                  attention_res=attention_res)

    @torch.no_grad()
    def ddim_sampling(self, shape, input, uc, uimage_back, guidance_scale=1, mask=None, x0=None,
                      attention_res=[8, 12]):
        b, n, _, _, _ = shape

        img = input["x"]
        if img == None:
            img = torch.randn(shape, device=self.device)
            input["x"] = img

        time_range = np.flip(self.ddim_timesteps)
        self.total_steps = self.ddim_timesteps.shape[0]

        # iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        iterator = time_range

        if self.alpha_generator_func != None:
            alphas = self.alpha_generator_func(len(iterator))

        for i, step in enumerate(iterator):

            # set alpha
            if self.alpha_generator_func != None:
                self.set_alpha_scale(self.model, alphas[i])
                if alphas[i] == 0:
                    self.model.restore_first_conv_from_SD()

            # run
            index = self.total_steps - i - 1
            input["timesteps"] = torch.full(
                (b,), step, device=self.device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.diffusion.q_sample(x0, input["timesteps"])
                img = img_orig * mask + (1. - mask) * img
                input["x"] = img

            img, pred_x0 = self.p_sample_ddim(i, input, index=index, uc=uc, uimage_back=uimage_back, guidance_scale=guidance_scale,
                                              attention_res=attention_res)                     
         
            input["x"] = img

        return img

    @torch.no_grad()
    def p_sample_ddim(self, i, input, index, uc=None, uimage_back=None, guidance_scale=1,
                      attention_res=[32, 48]):
        e_t = self.model(input) 
        if uc is not None and guidance_scale != 1:
            unconditional_input = dict(x=input["x"], timesteps=input["timesteps"], context=uc, image_back=uimage_back, grounding_extra_input=input['grounding_extra_input'])
            e_t_uncond = self.model( unconditional_input ) 
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)

        # select parameters corresponding to the currently considered timestep
        b, n, _, _, _ = input["x"].shape
        masks = input["grounding_input"]['masks'] 
        a_t = torch.full(
            (b, n, 1, 1, 1), self.ddim_alphas[index], device=self.device)
        a_prev = torch.full(
            (b, n, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full(
            (b, n, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full(
            (b, n, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=self.device)
        
        # self.aggregate_and_get_max_attention_per_token(
        #         attention_res=attention_res,  
        #         map_list=util._global_box_cross_atten_map,
        #         masks=masks,
        #         time_step=i
        # )

        # self.aggregate_context_attention(
        #         res=attention_res,
        #         select=0,
        #         time_step=i,
        #         map_list=util._global_context_cross_atten_map)                


        util.clean_value()

        # current prediction for x_0
        pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(input["x"])
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    def aggregate_and_get_max_attention_per_token(self,
                                                  attention_res: List[int] = [
                                                      8, 12],
                                                  map_list: List[torch.Tensor] = None,
                                                  masks: torch.Tensor = None,
                                                  time_step: int = 0
                                                  ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = self.aggregate_attention(
            res=attention_res,
            select=0,
            map_list=map_list)  

        self.compute_max_attention_per_index(
            attention_maps=attention_maps,
            masks=masks,
            time_step=time_step
        )
        

    def aggregate_attention(self, res: List[int],
                            select: int,
                            map_list: List[torch.Tensor]) -> torch.Tensor:
        """ Aggregates the attention across the different layers and heads at the specified resolution. """
        out = []
        num_pixels = res[0] * res[1]
        num_camera = self.model.num_camera

        for cross_atten_map in map_list:
            bn, N, M = cross_atten_map.shape       
            assert num_pixels == N
            cross_atten_map = rearrange(cross_atten_map, "(b n) N M -> b n N M ", n=num_camera)
            cross_atten_map = cross_atten_map.reshape(1, -1, num_camera, res[0], res[1], M)
            cross_atten_map = cross_atten_map[select]
            out.append(cross_atten_map)

        out = torch.stack(out)   # [10, 2, 1, 8, 12, 55]
        out = out.sum(0) / out.shape[0]  # [2, 1, 8, 12, 55]

        return out

    def aggregate_context_attention(self, res: List[int],
                            select: int,
                            map_list: List[torch.Tensor],
                            time_step:int,
                            ) -> torch.Tensor:
        """ Aggregates the attention across the different layers and heads at the specified resolution. """
        out = []
        num_pixels = res[0] * res[1]
        num_camera = self.model.num_camera

        for cross_atten_map in map_list:
            bn, N, M = cross_atten_map.shape       
            assert num_pixels == N

            # if not smooth_attentions:
            #     smoothing = GaussianSmoothing(
            #                 channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
            #     input = F.pad(cross_atten_map.unsqueeze(1),
            #                   (1, 1, 1, 1), mode='reflect')                

            #     cross_atten_map = smoothing(input).squeeze(1)

            cross_atten_map = rearrange(cross_atten_map, "(b n) N M -> b n N M ", n=num_camera)
            cross_atten_map = cross_atten_map.reshape(1, -1, num_camera, res[0], res[1], M)
            cross_atten_map = cross_atten_map[select]
            cross_atten_map = cross_atten_map.sum(-1) / cross_atten_map.shape[-1] # [b,n,h,w]
                        
            out.append(cross_atten_map)

        out = torch.stack(out)   # [10, 2, 1, 8, 12]
        out = out.sum(0) / out.shape[0]  # [2, 1, 8, 12]

        out *= 100 # [2, 1, 8, 12]
        out = torch.nn.functional.softmax(
            out, dim=-1)
        
        save_out = rearrange(out, "b n h w->(b n) h w").detach().cpu().numpy()
        bn = save_out.shape[0]
        if time_step == self.total_steps - 1 or time_step % 10 == 0:
            for b_i in range(bn):
                plt.figure(figsize=(12, 8))
                sns.heatmap(save_out[b_i], cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=False)
                plt.title(f"text {b_i+1}")
                save_path = os.path.join(
                                self.name, "context_" + str(self.iter_idx).zfill(3) + "_" + str(b_i + 1).zfill(2)  + "_T_" + str(time_step).zfill(3) + '.png')
                plt.savefig(save_path)
                self.writer.add_figure("Iter_{}_img_{}_context_T_{}".format(self.iter_idx, b_i + 1, time_step), plt.gcf(), global_step=b_i)                          
         

    def compute_max_attention_per_index(self,
                                        attention_maps: torch.Tensor,
                                        masks: torch.Tensor=None,
                                        time_step:int = 0
                                        ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """

        B, N, H, W, M = attention_maps.shape

        attention_maps *= 100 # [2, 1, 8, 12, 55]
        attention_maps = torch.nn.functional.softmax(
            attention_maps, dim=-1)

        average_attention_maps = attention_maps.sum(-1) / attention_maps.shape[-1]
        average_attention_maps = rearrange(average_attention_maps, "b n h w->(b n) h w").detach().cpu().numpy()

        bn = average_attention_maps.shape[0]
        if time_step == self.total_steps - 1 or time_step % 10 == 0:
            for b_i in range(bn):
                plt.figure(figsize=(12, 8))
                sns.heatmap(average_attention_maps[b_i], cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=False)
                plt.title(f"average {b_i + 1}")
                save_path = os.path.join(
                                self.name, "average_" + str(self.iter_idx).zfill(3) + "_img_" + str(b_i + 1).zfill(2) + "_T_" + str(time_step).zfill(3)+ '.png')
                plt.savefig(save_path)
                self.writer.add_figure("Iter_{}_img_{}_average_T_{}".format(self.iter_idx, b_i + 1, time_step), plt.gcf(), global_step=b_i)                                 


        for b_i in range(B):
            for n_i in range(N):
                per_cam_mask = masks[b_i, n_i, :]
                sum_boxes = int(per_cam_mask.sum())
                self.writer.add_scalar("Iter_{}_batch_{}_sum_boxes".format(self.iter_idx, b_i + 1), sum_boxes, n_i + 1)
                if sum_boxes == 0:
                    continue
                per_cam_bbox_map = []
                
                for box_i in range(sum_boxes):
                    bbox_map = attention_maps[b_i, n_i, :, :, box_i] # [H,W]
                    per_cam_bbox_map.append(bbox_map.float())

                    
                if time_step == self.total_steps - 1 or time_step % 10 == 0:
                    per_cam_bbox_map = torch.stack(per_cam_bbox_map) # [sum_bbox, H, W]
                    per_cam_bbox_map = per_cam_bbox_map.detach().cpu().numpy()
                    
                    for j in range(sum_boxes):
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(per_cam_bbox_map[j], cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=False)
                        plt.title(f"box {j+1}")
                        save_path = os.path.join(
                            self.name, "map_" + str(self.iter_idx).zfill(3) + "_img_" + str(b_i + 1).zfill(2) + "_cam_" + str(n_i + 1).zfill(1) + "_T_" + str(time_step).zfill(3) + "_box" + str(j + 1).zfill(2)+ '.png')
                        plt.savefig(save_path)
    

                        self.writer.add_figure("Iter_{}_img_{}_cam_{}_maps_T_{}".format(self.iter_idx, b_i + 1, n_i + 1, time_step), plt.gcf(), global_step= j + 1)                                              




