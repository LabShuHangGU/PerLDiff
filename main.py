# Standard library imports
import os
import json
import random
import argparse
from datetime import datetime

# Third-party imports
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import torch.distributed as dist
from omegaconf import OmegaConf
import gradio as gr

# Local application/library specific imports
from scripts.distributed import synchronize


def is_main_process():
    return  dist.get_rank() == 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_ROOT", type=str,  default="DATA", help="path to DATA")
    parser.add_argument("--OUTPUT_ROOT", type=str,  default="OUTPUT", help="path to OUTPUT")

    parser.add_argument("--name", type=str,  default="nusc_256x384_perldiff", help="experiment will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--seed", type=int,  default=123, help="used in sampler")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--yaml_file", type=str,  default="configs/nusc_text.yaml", help="paths to base configs.")


    parser.add_argument("--base_learning_rate", type=float,  default=5e-5, help="")
    parser.add_argument("--weight_decay", type=float,  default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int,  default=10000, help="")
    parser.add_argument("--scheduler_type", type=str,  default='constant', help="cosine or constant")
    parser.add_argument("--batch_size", type=int,  default=1, help="")
    parser.add_argument("--workers", type=int,  default=16, help="")
    parser.add_argument("--official_ckpt_name", type=str,  default="sd-v1-4.ckpt", help="SD ckpt name and it is expected in DATA_ROOT, thus DATA_ROOT/official_ckpt_name must exists")
    
    parser.add_argument('--enable_ema', default=False, type=lambda x:x.lower() == "true")
    parser.add_argument("--ema_rate", type=float,  default=0.9999, help="")
    parser.add_argument("--total_iters", type=int,  default=60000, help="")
    parser.add_argument("--save_every_iters", type=int,  default=6000, help="")
    parser.add_argument("--disable_inference_in_training", type=lambda x:x.lower() == "true",  default=False, help="Do not do inference, thus it is faster to run first a few iters. It may be useful for debugging ")
    parser.add_argument("--guidance_scale_c", type=float, default=None, help="guidance scale for classifier free guidance")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="extra digit, fewer digits, cropped, worst quality, low quality",
        help="",
    ) 

    parser.add_argument("--val_ckpt_name", type=str,  default="DATA/perldiff_256x384_lambda_5_bs2x8_model_checkpoint_00060000.pth", help="")
    
    parser.add_argument("--plms", action='store_true',
                        help="use plms sampling instead of ddim sampling")
    
    parser.add_argument("--step", type=int, default=50, help="ddim or plms sampling steps")
    parser.add_argument("--gen_path", type=str,  default="save path of samples_gen", help="")

    parser.add_argument("--training", action='store_true', help="train model")
    parser.add_argument("--validation", action='store_true', help=" test model")
    parser.add_argument("--generation", action='store_true', help=" generation model")

    parser.add_argument("--sample_index", type=int, default=-1, help="the data index to generate")

    args = parser.parse_args()
    assert args.scheduler_type in ['cosine', 'constant']

    num_gpus = torch.cuda.device_count()

    if num_gpus == 1:
        args.local_rank = 0
    else:
        args.rank = int(os.environ["RANK"]) 
        world_size = int(os.environ['WORLD_SIZE']) 
        args.local_rank = int(os.environ['LOCAL_RANK']) 

        torch.cuda.set_device(args.local_rank)
        dist_backend = 'nccl'
        args.dist_url = 'tcp://{}:{}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        print("args.dist_url:", args.dist_url)
        print('| distributed init (rank {}, word {}): {}'.format(args.rank, world_size, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=dist_backend, init_method=args.dist_url,
                                            world_size=world_size, rank=args.rank)
        
        torch.distributed.barrier()


    print(f"****************** ngpu={num_gpus} **************************")
    args.distributed = num_gpus > 1

    config = OmegaConf.load(args.yaml_file) 
    config.update( vars(args) )
    config.total_batch_size = config.batch_size * num_gpus
    print(f"****************** config.batch_size={config.batch_size} **************************")

    if args.training:
        from trainer import Trainer
        trainer = Trainer(config)
        synchronize()
        trainer.start_training()
    elif args.validation:
        from valer import Valer
        nuscene_valer = Valer(config)
        synchronize()
        nuscene_valer.start_validation()
    elif args.generation:
        from gener import Gener
        nuscene_gener = Gener(config)
        synchronize()
        nuscene_gener.start_validation(is_train=False) # False means validation, True means train
    else:
        NotImplementedError

    
