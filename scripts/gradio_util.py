# Required Python standard library imports
import os
import json
from datetime import datetime

from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
from dataset.utils import yaw_to_quaternion, quaternion_to_yaw, color_to_rgb, draw_box_3d

from einops import rearrange


def safe_filename(base_dir, prefix=""):
    """
    Generate safe file names to avoid overwriting existing files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{prefix}_{timestamp}.json"
    return os.path.join(base_dir, safe_name)


def limit_files(folder_path, max_files=10):
    """
    Limit the number of files in a folder. If max_files is exceeded, the oldest file is deleted.
    """
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    if len(files) > max_files:
        files.sort(key=lambda x: os.path.getmtime(x))
        
        for f in files[:-max_files]:
            os.remove(f)

def save_json(number, text_data):
    folder_path = "saved/jsons/"  
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    limit_files(folder_path)
    
    file_path = safe_filename(folder_path, "edited")
    if text_data.strip() != "":
        try:
            json_data = json.loads(text_data)
            with open(file_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)
            print(f"File saved: {file_path}")
            return file_path  
        except json.JSONDecodeError as e:
            print("Invalid JSON provided.")
            return None  
    return None
def check_class_name(class_name=None, CLASS_NAMES=None):
    '''
    '''
    if class_name in CLASS_NAMES:
        return True
    else:
        return False

def combine_images_into_grid(image_back_list, rows=2, cols=3, spacing=10):
    w, h = image_back_list[0].shape[1], image_back_list[0].shape[0]
    grid_width = w * cols + (cols - 1) * spacing
    grid_height = h * rows + (rows - 1) * spacing
    
    grid_img = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    for i, img_array in enumerate(image_back_list):
        img = Image.fromarray(img_array.astype('uint8')) 
        x_offset = (w + spacing) * (i % cols)
        y_offset = (h + spacing) * (i // cols)
        grid_img.paste(img, (x_offset, y_offset))
        
    return grid_img

def tensors_to_images_list(tensors):
    tensors = (rearrange(tensors, 'n c h w -> n h w c') * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    return [tensors[i] for i in range(tensors.shape[0])]


def vis_getitem_data(index=None, out=None, return_tensor=False, name="res.jpg", print_scene_description=True, normalize=False):

    if normalize:
        img = torchvision.transforms.functional.to_pil_image(
        out["image"])
    else:
        img = torchvision.transforms.functional.to_pil_image(
            out["image"]*0.5 + 0.5)
    canvas = torchvision.transforms.functional.to_pil_image(
        torch.ones_like(out["image"]))
    W, H = img.size

    if print_scene_description:
        scene_description = out["scene_description"]
        print(scene_description)
        print(" ")

    boxes_3d = []
    for box in out["box"]:
        # print(f"out {box.shape}")
        box = rearrange(box, '(n c)-> n c', n=8, c=2)
        boxes = []
        for i in range(8):
            x, y = box[i]
            boxes.append(torch.tensor([float(x*W), float(y*H)]))
        boxes = torch.stack(boxes)
        boxes_3d.append(boxes)
    boxes_3d = torch.stack(boxes_3d)
    img = draw_box_3d(img, boxes_3d)

    if return_tensor:
        return torchvision.transforms.functional.to_tensor(img)
    else:
        img.save(name)
 

