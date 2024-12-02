import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
import random
import time 
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
import shutil
import torchvision

import math
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from scripts.distributed import get_rank, synchronize, get_world_size
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy

from transformers import CLIPProcessor, CLIPModel
from einops import rearrange, repeat

from ldm.util import count_params

try:
    from apex import amp
except:
    pass  
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #

def read_official_convnext_ckpt(ckpt_path):      
    "Read offical pretrained convnext ckpt and convert into my style" 
    print( "\n" + "*" * 20 + " load model from {}!".format(ckpt_path) + " *" * 20 + "\n")

    state_dict = torch.load(ckpt_path, map_location="cpu")
  
    return state_dict

def read_official_gligen_ckpt(ckpt_path):      
    "Read offical pretrained GLIGEN ckpt and convert into my style" 
    print( "\n" + "*" * 20 + " load model from {}!".format(ckpt_path) + " *" * 20 + "\n")

    state_dict = torch.load(ckpt_path, map_location="cpu")

    for xname in list(state_dict['model'].keys()):
        if 'position_net' in xname or 'fuser' in xname:
            del  state_dict['model'][xname]
  
    return state_dict

def read_official_sd_ckpt(ckpt_path):      
    "Read offical pretrained SD ckpt and convert into my style" 
    print( "\n" + "*" * 20 + " load model from {}!".format(ckpt_path) + " *" * 20 + "\n")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     
    return out 


class ImageCaptionSaver:
    def __init__(self, base_path, nrow=6, normalize=True, scale_each=True, range=(0,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, images=None, real=None, foreground=None, background=None, captions=None, seen=None):
        # print("*"*30 + "save result" + "*"*30)
        batch_size, num_camera = images.shape[0], images.shape[1]
        if num_camera != 1:
            for i in range(batch_size):
                if images != None:
                    if not os.path.exists(os.path.join(self.base_path, 'gen')):
                        os.makedirs(os.path.join(self.base_path, 'gen'))
                    save_path = os.path.join(self.base_path, "gen", str(seen).zfill(8) + "_" + str(i) +'.png')
                    torchvision.utils.save_image( images[i], save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.range )
                
                if real != None:
                    if not os.path.exists(os.path.join(self.base_path, 'real')):
                        os.makedirs(os.path.join(self.base_path, 'real'))
                    save_path = os.path.join(self.base_path, "real", str(seen).zfill(8)+  "_real_" + str(i) +'.png')
                    torchvision.utils.save_image( real[i], save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.range)
                
                if foreground != None:
                    if not os.path.exists(os.path.join(self.base_path, 'foreground')):
                        os.makedirs(os.path.join(self.base_path, 'foreground'))
                    save_path = os.path.join(self.base_path, 'foreground', str(seen).zfill(8)+ "_foreground_" + str(i) + '.png')
                    torchvision.utils.save_image( foreground[i], save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.range)

                if background != None:
                    if not os.path.exists(os.path.join(self.base_path, 'background')):
                        os.makedirs(os.path.join(self.base_path, 'background'))
                    save_path = os.path.join(self.base_path, 'background', str(seen).zfill(8)+ "_background_" + str(i) + '.png')
                    torchvision.utils.save_image( background[i], save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.range)
        else:
            if images != None:
                if not os.path.exists(os.path.join(self.base_path, 'gen')):
                    os.makedirs(os.path.join(self.base_path, 'gen'))
                images = rearrange(images, 'b n c h w -> (b n) c h w')
                save_path = os.path.join(self.base_path, "gen", str(seen).zfill(8) + '_.png')
                torchvision.utils.save_image( images, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.range )
            
            if real != None:
                if not os.path.exists(os.path.join(self.base_path, 'real')):
                    os.makedirs(os.path.join(self.base_path, 'real'))
                real = rearrange(real, 'b n c h w -> (b n) c h w')
                save_path = os.path.join(self.base_path, "real", str(seen).zfill(8) +  '_real.png')
                torchvision.utils.save_image( real, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.range)
            if foreground != None:
                if not os.path.exists(os.path.join(self.base_path, 'foreground')):
                    os.makedirs(os.path.join(self.base_path, 'foreground'))
                foreground = rearrange(foreground, 'b n c h w -> (b n) c h w')
                save_path = os.path.join(self.base_path, "foreground", str(seen).zfill(8) + '_foreground.png')
                torchvision.utils.save_image( foreground, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.range)
            if background != None:
                if not os.path.exists(os.path.join(self.base_path, 'background')):
                    os.makedirs(os.path.join(self.base_path, 'background'))
                background = rearrange(background, 'b n c h w -> (b n) c h w')
                save_path = os.path.join(self.base_path, "background", str(seen).zfill(8) + '_background.png')
                torchvision.utils.save_image( background, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.range)



def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_sum_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print(f"total_trainable_params_count is: {total_trainable_params_count*1.e-6:.2f} M params")


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

           
def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    name = os.path.join( OUTPUT_ROOT, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_model_ckpt = os.path.join( name, previous_tag, "model", 'model_checkpoint_latest.pth' )
            if os.path.exists(potential_model_ckpt):
                checkpoint['model'] = potential_model_ckpt
                checkpoint['opt'] = os.path.join( name, previous_tag, "opt", 'opt_checkpoint_latest.pth' )
                checkpoint['scheduler'] = os.path.join( name, previous_tag, "scheduler",'scheduler_checkpoint_latest.pth' )

                potential_model_ema_ckpt = os.path.join( name, previous_tag, "ema", 'ema_checkpoint_latest.pth' )
                if os.path.exists(potential_model_ema_ckpt):
                    checkpoint['ema'] = potential_model_ema_ckpt

                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_model_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 






class Trainer:
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
        
        if self.config.official_ckpt_name == "sd-v1-4.ckpt":
            state_dict = read_official_sd_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
        else:
            state_dict = read_official_gligen_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
 
                
        missing_keys, unexpected_keys = self.model.load_state_dict( state_dict["model"], strict=False  )
        assert unexpected_keys == []
        original_params_names = list( state_dict["model"].keys()  ) # used for sanity check later 
        
        self.autoencoder.load_state_dict( state_dict["autoencoder"]  )
        self.text_encoder.load_state_dict( state_dict["text_encoder"]  )
        self.diffusion.load_state_dict( state_dict["diffusion"]  )
 
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


        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        params = []
        trainable_names = []
        all_params_name = []
        for name, p in self.model.named_parameters():
            if ("transformer_blocks" in name) and ("cross_view_left" in name):
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
            elif ("transformer_blocks" in name) and ("cross_view_right" in name):
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
            elif ("transformer_blocks" in name) and ("fuser" in name):
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
            elif ("transformer_blocks" in name) and ("attn_back" in name):
                params.append(p) 
                trainable_names.append(name)
            elif ("transformer_blocks" in name) and ("norm_back" in name):
                params.append(p) 
                trainable_names.append(name)
            elif ("transformer_blocks" in name) and ("attn2" in name):
                params.append(p) 
                trainable_names.append(name)
            elif  "position_net" in name:
                # Grounding token processing network 
                params.append(p) 
                trainable_names.append(name)
            else:
                # Following make sure we do not miss any new params
                # all new added trainable params have to be haddled above
                # otherwise it will trigger the following error  
                assert name in original_params_names, name 
            all_params_name.append(name) 


        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        
        if get_rank() == 0:
            count_sum_params(params)
        
        #  = = = = = EMA... It is worse than normal model in early experiments, thus never enabled later = = = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters()) 
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()




        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False 




        # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #  
        train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None
        dataset_train = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, train=True, repeats=train_dataset_repeats)
        # dataset_val = ConCatDataset(config.val_dataset_names, config.DATA_ROOT, train=False, repeats=None)
        
        sampler = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None 
        # sampler_val = DistributedSampler(dataset_val, seed=config.seed) if config.distributed else None 


        loader_train = DataLoader( dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=(sampler is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler)
        # loader_val = DataLoader( dataset_val,  batch_size=config.batch_size, 
        #                                            shuffle=False,
        #                                            num_workers=config.workers, 
        #                                            pin_memory=True, 
        #                                            sampler=sampler_val)

        self.dataset_train = dataset_train
        # self.dataset_val = dataset_val
        
        self.loader_train = wrap_loader(loader_train)
        # self.loader_val = wrap_loader(loader_val)

        if get_rank() == 0:
            total_image = dataset_train.total_images()
            # total_val_image = dataset_val.total_images()
            print("Total training images: ", total_image)
            # print("Total validation images: ", total_val_image)     



        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0   
        if checkpoint:
            checkpoint_model = torch.load(checkpoint['model'], map_location="cpu")
            checkpoint_opt = torch.load(checkpoint['opt'], map_location="cpu")
            checkpoint_scheduler = torch.load(checkpoint['scheduler'], map_location="cpu")
            
            self.model.load_state_dict(checkpoint_model['model'])
            if config.enable_ema:
                checkpoint_ema = torch.load(checkpoint['ema'], map_location="cpu")
                self.ema.load_state_dict(checkpoint_ema["ema"])
            
            self.opt.load_state_dict(checkpoint_opt['opt'])
            self.scheduler.load_state_dict(checkpoint_scheduler['scheduler'])
            self.starting_iter = checkpoint_model["iters"]
            if self.starting_iter >= config.total_iters:
                synchronize()
                print("Training finished. Start exiting")
                exit()

        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#    
        
        # func return input for controlling condition encoding
        self.controlling_condition_input = instantiate_from_config(config.controlling_condition_input)
        self.model.controlling_condition_input = self.controlling_condition_input
        
        if get_rank() == 0:       
            self.image_caption_saver = ImageCaptionSaver(self.name)

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

    def run_one_step(self, batch):
        x_start, t = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        controlling_condition_input = self.controlling_condition_input.prepare(batch)
        input = dict(x=x_noisy, 
                    timesteps=t, 
                    controlling_condition_input=controlling_condition_input)
        model_output = self.model(input)
        
        loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight

        self.loss_dict = {"loss": loss.item()}

        return loss 
        


    def start_training(self):

        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',  disable=get_rank() != 0 )
        self.model.train()
        for iter_idx in iterator: # note: iter_idx is not from 0 if resume training
            self.iter_idx = iter_idx

            self.opt.zero_grad()
            batch = next(self.loader_train)
            
            batch_to_device(batch, self.device)
            with torch.no_grad():
                batch = self.apdate_batch(batch)
            loss = self.run_one_step(batch)
            loss.backward()
            self.opt.step() 
            self.scheduler.step()
            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)


            if (get_rank() == 0):
                if (iter_idx % 10 == 0):
                    self.log_loss()
                    print(f"Training progress (iter: {iter_idx}, loss: {loss.item():.4f})") 
                    
                if (iter_idx == 0)  or  ( iter_idx % self.config.save_every_iters == 0 ):
                    self.save_ckpt_and_result(save_ckpt=True)
                if (iter_idx == self.config.total_iters-1):
                    self.save_ckpt_and_result(save_ckpt=True)
            synchronize()

        synchronize()
        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    

    @torch.no_grad()
    def save_ckpt_and_result(self, save_ckpt=False):

        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if not self.config.disable_inference_in_training:
            # Do an inference on one training batch 
            batch_here = self.config.batch_size
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)
            with torch.no_grad():
                batch = self.apdate_batch(batch)

            num_cam = batch['image'].shape[1]

            real_images_with_box_drawing = [] # we save this durining trianing for better visualization
            for i in range(batch_here):
                images_per_cam = []
                for j in range(num_cam):
                    temp_data = {"image": batch["image"][i][j], "box":batch["box"][i][j]}
                    im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_scene_description=False)
                    images_per_cam.append(im)
                images_per_cam = torch.stack(images_per_cam)
                real_images_with_box_drawing.append(images_per_cam)
            real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)


            if self.config.plms:
                sampler = PLMSSampler(self.diffusion, model_wo_wrapper)
            else:
                sampler = DDIMSampler(self.diffusion, model_wo_wrapper)
                   
            shape = (batch_here, model_wo_wrapper.num_camera, model_wo_wrapper.in_channels, model_wo_wrapper.image_size[0], model_wo_wrapper.image_size[1])
            
            controlling_condition_input = self.controlling_condition_input.prepare(batch)
            input = dict( x=None, 
                          timesteps=None, 
                          controlling_condition_input=controlling_condition_input )
           
            samples = sampler.sample(S=self.config.step, shape=shape, input=input, guidance_scale_c=self.config.guidance_scale_c)
            
            autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
            samples = autoencoder_wo_wrapper.decode(samples).cpu()
            samples = torch.clamp(samples, min=-1, max=1)

            sample_images_with_box_drawing = [] # we save this durining trianing for better visualization
            for i in range(batch_here):
                images_per_cam = []
                for j in range(num_cam):
                    temp_data = {"image": samples[i][j], "box":batch["box"][i][j]}
                    im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_scene_description=False)
                    images_per_cam.append(im)
                images_per_cam = torch.stack(images_per_cam)
                sample_images_with_box_drawing.append(images_per_cam)
            sample_images_with_box_drawing = torch.stack(sample_images_with_box_drawing)

            self.image_caption_saver(images=sample_images_with_box_drawing, real=real_images_with_box_drawing,  foreground=None,seen=iter_name)
            
        if save_ckpt:

            model_ckpt = dict(model = model_wo_wrapper.state_dict(),
                        iters = self.iter_idx+1
            )
            opt_ckpt = dict(
                        opt= self.opt.state_dict(),
                        iters = self.iter_idx+1
            )
            scheduler_ckpt = dict(
                        scheduler= self.scheduler.state_dict(),
                        iters = self.iter_idx+1
            )
            temp_dirs = ["model", "opt", "scheduler"]
            if self.config.enable_ema:
                model_ema_ckpt = dict(
                    ema = self.ema.state_dict(),
                    iter= self.iter_idx+1
                )
                temp_dirs.append("ema")
            
            for dir in temp_dirs:
                dirs = os.path.join(self.name, dir)
                if not os.path.exists(dirs):
                    os.makedirs(dirs)

            torch.save( model_ckpt, os.path.join(self.name, "model", "model_checkpoint_"+str(iter_name).zfill(8)+".pth") )
            torch.save( model_ckpt, os.path.join(self.name, "model", "model_checkpoint_latest.pth") )

            torch.save( opt_ckpt, os.path.join(self.name, "opt", "opt_checkpoint_"+str(iter_name).zfill(8)+".pth") )
            torch.save( opt_ckpt, os.path.join(self.name, "opt", "opt_checkpoint_latest.pth") )

            torch.save( scheduler_ckpt, os.path.join(self.name, "scheduler", "scheduler_checkpoint_"+str(iter_name).zfill(8)+".pth") )
            torch.save( scheduler_ckpt, os.path.join(self.name, "scheduler", "scheduler_checkpoint_latest.pth") )
            
            if self.config.enable_ema:
                torch.save( model_ema_ckpt, os.path.join(self.name, "ema", "ema_checkpoint_"+str(iter_name).zfill(8)+".pth") )
                torch.save( model_ema_ckpt, os.path.join(self.name, "ema", "ema_checkpoint_latest.pth") )



