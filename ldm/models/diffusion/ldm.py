import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from ldm.util import default
from ldm.modules.diffusionmodules.util import  extract_into_tensor
from ldm.models.diffusion.ddpm import DDPM



class LatentDiffusion(DDPM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # hardcoded 
        self.clip_denoised = False
        


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    def reverse_q_sample(self, x_noisy, t, noise=None):
        return (x_noisy - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape) * noise) / \
        extract_into_tensor(self.sqrt_alphas_cumprod, t, x_noisy.shape)

    def reverse_q_sample(self, x_noisy, t, noise=None):
        return (x_noisy - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape) * noise) / \
        extract_into_tensor(self.sqrt_alphas_cumprod, t, x_noisy.shape)


    "Does not support DDPM sampling anymore. Only do DDIM or PLMS"






