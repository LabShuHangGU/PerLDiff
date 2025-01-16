import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

from ldm import util 
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
# import seaborn as sns
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from einops import rearrange, repeat
import torch.nn.functional as F
from gaussian_smothing import GaussianSmoothing
import torchvision

class EDMSampler(object):
    def __init__(self, diffusion, model, autoencoder=None, schedule="linear", alpha_generator_func=None, set_alpha_scale=None, name=None, iter_idx=None, x_start_save_interval=None):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.device = diffusion.betas.device
        self.ddpm_num_timesteps = diffusion.num_timesteps
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func # func
        self.set_alpha_scale = set_alpha_scale # func

        self.name = name
        self.iter_idx = iter_idx

        self.autoencoder = autoencoder
        self.x_start_save_interval = x_start_save_interval

        # Using the boxdiffusion method
        util._init_()

        if not os.path.exists(os.path.join(self.name, 'Log')):
            os.makedirs(os.path.join(self.name, 'Log'))

        if not os.path.exists(os.path.join(self.name, 'Map')):
            os.makedirs(os.path.join(self.name, 'Map'))

        self.save_attn_map_path = os.path.join(self.name, 'Map')
        
        self.writer = SummaryWriter(os.path.join(self.name, 'Log'))
        

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)


    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=False)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=False)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    @torch.no_grad()
    def sample(self, S, shape, input, uc=None, guidance_scale_c=None,guidance_scale_s=None, mask=None, x0=None, sigma_min=0.002, sigma_max=80, rho=7,S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        self.make_schedule(ddim_num_steps=S)
        return self.ddim_sampling(shape, input, uc, guidance_scale_c, guidance_scale_s,  mask=mask, x0=x0, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise)
 

    @torch.no_grad()
    def ddim_sampling(self, shape, input, uc, guidance_scale_c=None, guidance_scale_s=None, mask=None, x0=None, sigma_min=0.002, sigma_max=80, rho=7,S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        b, n, _, _, _ = shape
        
        img = input["x"]
        if img == None:     
            img = torch.randn(shape, device=self.device)
            input["x"] = img

        # Adjust noise levels based on what's supported by the network.

        # Time step discretization.
        num_steps = self.ddim_timesteps.shape[0]
        step_indices = torch.arange(num_steps, dtype=img.dtype, device=self.device)
        # step_indices = torch.tensor(self.ddim_timesteps, device=self.device)
        
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0


        # Main sampling loop. 
        x_next = input["x"] * t_steps[0]
        for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
            
            # select parameters corresponding to the currently considered timestep
            index = num_steps - i - 1
            a_t = torch.full((b, n, 1, 1, 1), self.ddim_alphas[index], device=self.device)
            a_prev = torch.full((b, n, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
            sigma_t = torch.full((b, n, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
            sqrt_one_minus_at = torch.full((b, n, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)
            
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            input["timesteps"] = torch.full((b,), t_hat, device=self.device, dtype=torch.long)
            input["x"] = x_hat
            e_t = self.model(input) 

            if guidance_scale_c == None:
                raise ValueError("guidance_scale_c should not be None.")

            if uc is not None and guidance_scale_c != None:
                unconditional_input = dict(x=x_hat, timesteps=input["timesteps"], context=uc)
                e_t_uncond = self.model( unconditional_input ) 
                e_t = e_t_uncond + guidance_scale_c * (e_t - e_t_uncond)

            pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * torch.randn_like( input["x"] ) 
            denoised = a_prev.sqrt() * pred_x0 + dir_xt + noise

            # denoised = self.autoencoder.decode(x_prev)

            d_cur = (x_hat - denoised) / t_hat

            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                input["timesteps"] = torch.full((b,), t_next, device=self.device, dtype=torch.long)
                input["x"] = x_next
                e_t = self.model(input) 

                if uc is not None and guidance_scale_c != None:
                    unconditional_input = dict(x=input["x"], timesteps=input["timesteps"], context=uc)
                    e_t_uncond = self.model( unconditional_input ) 
                    e_t = e_t_uncond + guidance_scale_c * (e_t - e_t_uncond)

                pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()
                dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                noise = sigma_t * torch.randn_like( input["x"] ) 
                denoised = a_prev.sqrt() * pred_x0 + dir_xt + noise

                # denoised = self.autoencoder.decode(x_prev)
                
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next


    @torch.no_grad()
    def p_sample_ddim(self, input, index, uc=None, guidance_scale_c=None, guidance_scale_s=None, step=None, kernel_size=3, gaussian_sigma=3):
        b, n, c, h, w = input["x"].shape
        
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, n, 1, 1, 1), self.ddim_alphas[index], device=self.device)
        a_prev = torch.full((b, n, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, n, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, n, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)
        
        
        e_t = self.model(input) 

        if guidance_scale_c == None:
            raise ValueError("guidance_scale_c should not be None.")


        if uc is not None and guidance_scale_c != None:

            # pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # z_t_pred = a_t.sqrt() * (1-fore_masks) * pred_x0 + sqrt_one_minus_at * e_t

            unconditional_input = dict(x=input["x"], timesteps=input["timesteps"], context=uc)
            # unconditional_input = dict(x=z_t_pred, timesteps=input["timesteps"], context=uc)
            e_t_uncond = self.model( unconditional_input ) 

            # atten_maps = self.get_clear_attention_maps(self.model)
            e_t = e_t_uncond + guidance_scale_c * (e_t - e_t_uncond)

        # atten_maps = rearrange(atten_maps, "(b n) h w->b n h w", b=b,n=n)

        
        # ########## for visualizer attentionmap ###########
        # masks = input["grounding_input"]['masks'] 
        # self.aggregate_and_get_max_attention_per_token(
        #         attention_res=[32, 48],  
        #         map_list=util._global_box_cross_atten_map,
        #         masks=masks,
        #         time_step=step
        # )
        # ########### for visualizer attentionmap ###########
        util.clean_value()
        # current prediction for x_0
        pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # if guidance_scale_s != None:
        #     if step % 10 == 0:
        #         save_base_path = os.path.join(self.name, 'attention_map',f"iter_{self.iter_idx}_T_{step}.png")
        #         self.plot_batch_attention_maps(atten_maps, save_base_path,cmap="plasma") # plasma magma
        #         save_base_path = os.path.join(self.name, 'fore_masks',f"iter_{self.iter_idx}_T_{step}.png")
        #         self.plot_batch_attention_maps(fore_masks.detach().cpu(), save_base_path,cmap="plasma") # plasma magma

        #     # masks = repeat(atten_maps, "b n h w->b n c h w", c=c).to(pred_x0.device)
        #     masks = repeat(fore_masks, "b n h w->b n c h w", c=c).to(pred_x0.device)
            
        #     pred_x0_blur = rearrange(pred_x0,"b n c h w->(b n) c h w")
        #     smoothing = GaussianSmoothing(
        #                     channels=c, kernel_size=kernel_size, sigma=gaussian_sigma, dim=2).to(pred_x0.device)
            
        #     pred_x0_blur = F.pad(pred_x0_blur,
        #                       (1, 1, 1, 1), mode='reflect')  
        #     pred_x0_blur = smoothing( pred_x0_blur)
        #     pred_x0_blur = rearrange(pred_x0_blur, "(b n) c h w->b n c h w", b=b,n=n)

        #     x_t_blur = a_t.sqrt() * pred_x0_blur + sqrt_one_minus_at * e_t

        #     x_t_uncond  = (1- masks) * input["x"] + masks * x_t_blur

        #     unconditional_input_blur = dict(x=x_t_uncond, timesteps=input["timesteps"], context=uc)
        #     e_t_blur = self.model(unconditional_input_blur)

        #     e_t = e_t + guidance_scale_s * (e_t_uncond - e_t_blur)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * torch.randn_like( input["x"] ) 

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0
    
    @torch.no_grad()
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
            map_list=map_list)  

        self.compute_max_attention_per_index(
            attention_maps=attention_maps,
            masks=masks,
            time_step=time_step
        )
        

    def aggregate_attention(self, res: List[int],
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
            cross_atten_map = cross_atten_map[0]
            out.append(cross_atten_map)

        out = torch.stack(out)   # [10, b,n,h,w,m]
        out = out.sum(0) / out.shape[0]  # [b,n,h,w,m]

        return out
                    
         
    @torch.no_grad()
    def compute_max_attention_per_index(self,
                                        attention_maps: torch.Tensor,
                                        masks: torch.Tensor=None,
                                        time_step:int = 0
                                        ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """

        B, N, H, W, M = attention_maps.shape
                                
        for b_i in range(B):
            for n_i in range(N):
                per_cam_mask = masks[b_i, n_i, :]
                sum_boxes = int(per_cam_mask.sum())
                if sum_boxes == 0:
                    continue
                per_cam_bbox_map = []
                
                for box_i in range(sum_boxes):
                    bbox_map = attention_maps[b_i, n_i, :, :, box_i] # [H,W]
                    per_cam_bbox_map.append(bbox_map.float())

                    
                if time_step == self.total_steps - 1 or (time_step + 1) % 25 == 0:
                    per_cam_bbox_map = torch.stack(per_cam_bbox_map) # [sum_bbox, H, W]
                    per_cam_bbox_map = per_cam_bbox_map.detach().cpu().numpy()
                    
                    for j in range(sum_boxes):
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(per_cam_bbox_map[j], cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=False)
                        plt.title(f"box {j+1}")
                        save_path = os.path.join(
                            self.save_attn_map_path, str(self.iter_idx).zfill(2) + "_map_"  + "batch_" + str(b_i + 1).zfill(2) + "_cam_" + str(n_i + 1).zfill(1) + "_T_" + str(time_step + 1).zfill(3) + "_box_" + str(j + 1).zfill(2)+ '.png')
                        plt.savefig(save_path)



    def get_clear_attention_maps(self, model):
        from ldm.modules.attention import CrossAttention
        atten_maps = []
        for module in model.modules():
            if hasattr(module, 'atten_map') and isinstance(module, CrossAttention):
                # get atten_map
                atten_maps.append(module.atten_map[0]) # [B, Head, H, W]
                module.clear_attention_maps()
        atten_maps = torch.stack(atten_maps, dim=-1) # [B, Head, H, W, num_layer]

        B, Head, H, W, num_layer = atten_maps.shape

        # view_atten_maps = atten_maps[:,0,:,:,:].view(B, H*W, num_layer)
        view_atten_maps = atten_maps[:,:,:,:,-1].view(B, H*W, Head)

        # pooled_atten_maps = F.adaptive_avg_pool1d(view_atten_maps, 1).squeeze(-1).view(B, H, W)
        pooled_atten_maps = view_atten_maps[:, :, -1].view(B, H, W)
        return pooled_atten_maps
    



    def plot_batch_attention_maps(self, attn_maps, save_attn_map_path=None, max_batches=None, max_heads=None, close_previous=False, cmap='viridis'):
        """
        Plots and saves heatmaps for a batch of attention maps with multiple heads.
        
        Parameters:
        - attn_maps: A numpy array of attention maps with shape (B, N, H, W).
        - save_attn_map_path: Path to save the attention maps.
        - max_batches: An optional integer to limit the number of batches plotted.
        - max_heads: An optional integer to limit the number of heads plotted per batch.
        - close_previous: A boolean indicating whether to close previous figures.
        - cmap: The colormap for the heatmaps.
        """
        if close_previous:
            plt.close('all')  # Close all previous figures
        num_batches = attn_maps.shape[0] if max_batches is None else min(max_batches, attn_maps.shape[0])
        num_heads = attn_maps.shape[1] if max_heads is None else min(max_heads, attn_maps.shape[1])
        
        for b in range(num_batches):
            for n in range(num_heads):
                plt.figure(figsize=(32, 48))  # Smaller figure size for individual attention maps
                attention = attn_maps[b, n, :, :]
                cax = plt.matshow(attention, cmap=cmap)  # Plotting the heatmap with the specified colormap
                plt.colorbar(cax)  # Displaying color bar
                plt.title(f'Batch {b+1}, cam {n+1}')  # Adding a title to the figure
                plt.xticks([]), plt.yticks([])  # Removing axis labels
                
                if save_attn_map_path:
                    # Construct a unique filename for each batch and head
                    file_name = f'batch_{b+1}_cam_{n+1}.png'
                    full_path = os.path.join(save_attn_map_path, file_name)
                    
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    
                    plt.savefig(full_path)  # Save the figure to the specified path
                
                plt.close()  # Close the figure to save memory"








