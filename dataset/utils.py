#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import PIL
import torch
import torchvision.transforms as T

import cv2
import numpy as np


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


CLASSNAME_TO_COLOR = {  # RGB.
        "pedestrian": (135, 206, 235),  # Skyblue,
        "barrier": (112, 128, 144),  # Slategrey
        "traffic cone": (47, 79, 79),  # Darkslategrey
        "bicycle": (188, 143, 143),  # Rosybrown
        "bus": (255, 127, 80),  # Coral
        "car": (255, 158, 0),  # Orange
        "construction vehicle": (233, 150, 70),  # Darksalmon
        "motorcycle": (255, 61, 99),  # Red
        "trailer": (255, 140, 0),  # Darkorange
        "truck": (255, 99, 71),  # Tomato
    }


def imagenet_preprocess():
  return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def rescale(x):
  lo, hi = x.min(), x.max()
  return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
  transforms = [
    T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
    T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
  ]
  if rescale_image:
    transforms.append(rescale)
  return T.Compose(transforms)


def imagenet_deprocess_batch(imgs, rescale=True):
  """
  Input:
  - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

  Output:
  - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
    in the range [0, 255]
  """
  if isinstance(imgs, torch.autograd.Variable):
    imgs = imgs.data
  imgs = imgs.cpu().clone()
  deprocess_fn = imagenet_deprocess(rescale_image=rescale)
  imgs_de = []
  for i in range(imgs.size(0)):
    img_de = deprocess_fn(imgs[i])[None]
    img_de = img_de.mul(255).clamp(0, 255).byte()
    imgs_de.append(img_de)
  imgs_de = torch.cat(imgs_de, dim=0)
  return imgs_de


class Resize(object):
  def __init__(self, size, interp=PIL.Image.BILINEAR):
    if isinstance(size, tuple):
      H, W = size
      self.size = (W, H)
    else:
      self.size = (size, size)
    self.interp = interp

  def __call__(self, img):
    return img.resize(self.size, self.interp)


def unpack_var(v):
  if isinstance(v, torch.autograd.Variable):
    return v.data
  return v


def split_graph_batch(triples, obj_data, obj_to_img, triple_to_img):
  triples = unpack_var(triples)
  obj_data = [unpack_var(o) for o in obj_data]
  obj_to_img = unpack_var(obj_to_img)
  triple_to_img = unpack_var(triple_to_img)

  triples_out = []
  obj_data_out = [[] for _ in obj_data]
  obj_offset = 0
  N = obj_to_img.max() + 1
  for i in range(N):
    o_idxs = (obj_to_img == i).nonzero().view(-1)
    t_idxs = (triple_to_img == i).nonzero().view(-1)

    cur_triples = triples[t_idxs].clone()
    cur_triples[:, 0] -= obj_offset
    cur_triples[:, 2] -= obj_offset
    triples_out.append(cur_triples)

    for j, o_data in enumerate(obj_data):
      cur_o_data = None
      if o_data is not None:
        cur_o_data = o_data[o_idxs]
      obj_data_out[j].append(cur_o_data)

    obj_offset += o_idxs.size(0)

  return triples_out, obj_data_out



def yaw_to_quaternion(yaw):
    yaw_rad = yaw * np.pi / 180
    q = np.array([np.cos(yaw_rad / 2), 0, 0, np.sin(yaw_rad / 2)])
    return q


def quaternion_to_yaw(q):
    yaw = np.arctan2(2 * (q[0] * q[3] - q[1] * q[2]),
                     1 - 2 * (q[2] ** 2 + q[3] ** 2))
    yaw = yaw * 180 / np.pi
    return yaw


def color_to_rgb(color):
    color_map = {
        "red": (255, 0, 0),
        "olive": (128, 128, 0),
        "blue": (0, 0, 255),
        "green": (0, 128, 0),
        "orange": (255, 165, 0),
        "brown": (165, 42, 42),
        "cyan": (0, 255, 255),
        "purple": (128, 0, 128),
        
    }
    return color_map.get(color.lower())


def draw_box_3d(image, boxes, color=None):
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    image = np.array(image)
    for bid, box_img in enumerate(boxes):
        box_img = np.array(box_img).T.copy().astype(np.int32)
        # color = colors[bid % len(colors)]
        # if color == None:
        color = color_to_rgb(colors[4])

        for i in range(4):
            j = (i + 1) % 4
            cv2.line(image, (box_img[0, i], box_img[1, i]), (box_img[0, j], box_img[1, j]), color, thickness=1)
            cv2.line(image, (box_img[0, i + 4], box_img[1, i + 4]), (box_img[0, j + 4], box_img[1, j + 4]), color, thickness=1)
            cv2.line(image, (box_img[0, i], box_img[1, i]), (box_img[0, i + 4], box_img[1, i + 4]), color, thickness=1)
        
        center_point = (box_img[:, 2] + box_img[:,3] + box_img[:,6] + box_img[:, 7]) // 4

        fc_x = (box_img[0][2] + box_img[0][3]) // 2
        fc_y = (box_img[1][2] + box_img[1][3]) // 2

        cv2.line(image, center_point , (fc_x, fc_y) , color, thickness=2)
    return image

def get_color(class_name=None):
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """
    if class_name in CLASSNAME_TO_COLOR:
        return CLASSNAME_TO_COLOR[class_name]
    return None

def quaternion_multiply(quaternion1, quaternion2):
    """return quaternion multiplication of two quaternions。"""
    w0, x0, y0, z0 = quaternion1
    w1, x1, y1, z1 = quaternion2
    return np.array([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ])