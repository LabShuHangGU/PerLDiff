# Standard library imports
import os
import json
import random
import sys
import argparse
from datetime import datetime

# Third-party imports
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import torch.distributed as dist
from omegaconf import OmegaConf
import gradio as gr

sys.path.append('.')
# Local application/library specific imports
from scripts.distributed import synchronize
from scripts.vis_nusc_sample_info import Visualizer
from scripts.gradio_util import save_json, combine_images_into_grid, tensors_to_images_list


def is_main_process():
    return  dist.get_rank() == 0

# if __name__ == "__main__":

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_ROOT", type=str,  default="DATA", help="path to DATA")
parser.add_argument("--OUTPUT_ROOT", type=str,  default="OUTPUT", help="path to OUTPUT")

parser.add_argument("--name", type=str,  default="nusc_test_256x384_perldiff", help="experiment will be stored in OUTPUT_ROOT/name")
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


args = parser.parse_args()
assert args.scheduler_type in ['cosine', 'constant']

num_gpus = torch.cuda.device_count()



print(f"****************** ngpu={num_gpus} **************************")

config = OmegaConf.load(args.yaml_file) 
config.update( vars(args) )
config.total_batch_size = config.batch_size * num_gpus
print(f"****************** config.batch_size={config.batch_size} **************************")

valer = Visualizer(config)
samples_json_files = {}
for i in range(20):
    samples_json_files[str(i)] = f'samples_{i}_info.json'
synchronize()

def load_json(number, nusc_info_file_path="nuscenes_sample_info_files"):
    file_path = samples_json_files.get(str(number), None)
    file_path = os.path.join(nusc_info_file_path, file_path)
    if file_path and os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            data = json.dumps(json.load(json_file), indent=2)
            return data, file_path  # Return both the JSON content and the file path
    return "JSON file not found.", None  # Return None as the file path when not found

def get_road_map_and_image_gt(json_file_path=None, rot=0.0, pos=0.0, size=0.0, road_map_flag=False):
    
    road_map, image_gt = valer.prepare_road_map_with_draw(json_file_path=json_file_path, rot=rot, pos=pos, size=size)
    if road_map_flag:
        return road_map
    else:
        return image_gt

def show_selected_road_map(json_file_path=None, rot=0.0, pos=0.0, size=0.0):

    print(f"json_file_path = {json_file_path}")
    road_map = get_road_map_and_image_gt(json_file_path=json_file_path, rot=rot, pos=pos, size=size, road_map_flag=True)
    road_map_list = tensors_to_images_list(road_map)
    grid_image = combine_images_into_grid(road_map_list)

    return grid_image

def show_selected_image_gt(json_file_path=None, rot=0.0, pos=0.0, size=0.0):
    image_gt = get_road_map_and_image_gt(json_file_path=json_file_path, rot=rot, pos=pos, size=size)
    print(f"image_gt: {image_gt.shape}")
    image_gt_list = tensors_to_images_list(image_gt)
    grid_image = combine_images_into_grid(image_gt_list)

    return grid_image


def show_generated_images(json_file_path=None, rot=0.0, pos=0.0, size=0.0, scene_description="Realistic autonomous driving scene, day.", guidance_scale_c=5.0, step=50):
    generated_images = valer.start_vis(json_file_path=json_file_path, rot=rot, pos=pos, size=size, scene_description=scene_description, guidance_scale_c=guidance_scale_c, step=step)
    print(f"generated_images: {generated_images.shape}")
    generated_image_list = tensors_to_images_list(generated_images)
    grid_image = combine_images_into_grid(generated_image_list)

    return grid_image

def generate_and_show(json_file_path=None, rot=0.0, pos=0.0, size=0.0, weather=None, guidance_scale_c=5.0, step=50):
    if weather is None: 
        return "Please select a weather condition."
    if json_file_path is None or not os.path.exists(json_file_path):
        return "Please load a .json file first."
    if "day" in weather:
        weather = "day"
    elif "rain" in weather:
        weather = "rain"
    elif "night" in weather:
        weather = "night"
    scene_description = f"Realistic autonomous driving scene, {weather}."
    return show_generated_images(json_file_path, rot=rot, pos=pos, size=size, scene_description=scene_description, guidance_scale_c=guidance_scale_c, step=step)

def update_image_display(btn_pressed, *args, **kwargs):
    if btn_pressed:
        return generate_and_show(*args, **kwargs)
    return None  


with gr.Blocks() as demo:
    with gr.Row():
        # Dropdown to select the JSON file.
        number_input = gr.Dropdown(label="Select the scene file to load", choices=list(samples_json_files.keys()))
        load_json_button = gr.Button("Loading scene files")
        save_json_button = gr.Button("Save scene file")
    # Now, json_text will be used to store the JSON content, and the file path is directly passed.
    json_text, loaded_file_path = gr.Textbox(label="JSON file content", lines=20, placeholder="After loading, the scene file content is displayed and can be modified"), gr.Textbox(visible=False)
    load_json_button.click(fn=load_json, inputs=[number_input], outputs=[json_text, loaded_file_path])
    save_json_button.click(fn=save_json, inputs=[number_input, json_text], outputs=loaded_file_path)

    with gr.Row():
        rot = gr.Slider(label="Controlling the angle of an object", minimum=-180.0, maximum=180.0, value=0, step=10)
        pos = gr.Slider(label="Controlling the position of an object", minimum=-10.0, maximum=10.0, value=0.0, step=0.5)
        size = gr.Slider(label="Controlling object size", minimum=0.0, maximum=2.0, value=0.0, step=0.1)

    with gr.Row():
        weather_btn_group = gr.Radio(
            label="Select Weather/Time",
            choices=["rain", "day", "night"],
            value=None
        )
        ddim_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=100, value=50, step=1)
        scale = gr.Slider(label="Guidance scale", minimum=0.1, maximum=30.0, value=5.0, step=0.1)
    
    with gr.Row():
        submit_back_button = gr.Button("Visualization scene background image")
    combined_road_map_display = gr.Image(label="Multi-view background image")
    submit_back_button.click(
            fn=show_selected_road_map,
            inputs=[loaded_file_path, rot, pos, size],
            outputs=combined_road_map_display
        )

    with gr.Row():
        generate_button = gr.Button("Visual scene generation graph")
    
    generated_image_display = gr.Image(label="Multi-view generation graph")
    generate_button.click(
            fn=update_image_display, 
            inputs=[generate_button, loaded_file_path, rot, pos, size, weather_btn_group, scale, ddim_steps],
            outputs=generated_image_display
        )
    with gr.Row():
        submit_gt_button = gr.Button("Visualization scene real map")
    
    combined_image_gt_display = gr.Image(label="Multi-view real image")
    
    submit_gt_button.click(
        fn=show_selected_image_gt,
        inputs=[loaded_file_path, rot, pos, size],
        outputs=combined_image_gt_display
    )
demo.launch(server_name='0.0.0.0', server_port=1235, share=False)
    