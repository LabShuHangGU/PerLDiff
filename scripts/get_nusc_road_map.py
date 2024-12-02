# Standard library imports
import os
import sys
import math
import random
import json
from glob import glob
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Union

# Third-party imports
import descartes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset, DistributedSampler
from einops import rearrange, repeat
from transformers import CLIPProcessor, CLIPModel
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm

# NuScenes specific imports
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords

# Local application/library specific imports
sys.path.append('.')
from dataset.utils import yaw_to_quaternion, quaternion_to_yaw, color_to_rgb, draw_box_3d

device = torch.device("cuda")
locations = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']

class PerlDiff_NuscenesMap(NuScenesMap):
    def __init__(self,
                 dataroot: str = '/data/sets/nuscenes',
                 map_name: str = 'singapore-onenorth'):
        """
        Loads the layers, create reverse indices and shortcuts, initializes the explorer class.
        :param dataroot: Path to the layers in the form of a .json file.
        :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown`,
        `boston-seaport` that we want to load.
        """
        assert map_name in locations, 'Error: Unknown map name %s!' % map_name

        self.dataroot = dataroot
        self.map_name = map_name

        self.geometric_layers = ['polygon', 'line', 'node']

        # These are the non-geometric layers which have polygons as the geometric descriptors.
        self.non_geometric_polygon_layers = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                                             'walkway', 'stop_line', 'carpark_area']

        # We want to be able to search for lane connectors, but not render them.
        self.lookup_polygon_layers = self.non_geometric_polygon_layers + ['lane_connector']

        # These are the non-geometric layers which have line strings as the geometric descriptors.
        self.non_geometric_line_layers = ['road_divider', 'lane_divider', 'traffic_light']
        self.non_geometric_layers = self.non_geometric_polygon_layers + self.non_geometric_line_layers
        self.layer_names = self.geometric_layers + self.lookup_polygon_layers + self.non_geometric_line_layers

        # Load the selected map.
        self.json_fname = os.path.join(self.dataroot, 'maps', 'expansion', '{}.json'.format(self.map_name))
        with open(self.json_fname, 'r') as fh:
            self.json_obj = json.load(fh)

        # Parse the map version and print an error for deprecated maps.
        if 'version' in self.json_obj:
            self.version = self.json_obj['version']
        else:
            self.version = '1.0'
        if self.version < '1.3':
            raise Exception('Error: You are using an outdated map version (%s)! '
                            'Please go to https://www.nuscenes.org/download to download the latest map!')

        self.canvas_edge = self.json_obj['canvas_edge']
        self._load_layers()
        self._make_token2ind()
        self._make_shortcuts()

        self.explorer = PerlDiff_NuScenesMapExplorer(self)
    def render_map_wo_image(self,
                            nusc: NuScenes,
                            sample_token: str,
                            camera_channel: str = 'CAM_FRONT',
                            alpha: float = 0.3,
                            patch_radius: float = 10000,
                            min_polygon_area: float = 1000,
                            render_behind_cam: bool = True,
                            render_outside_im: bool = True,
                            layer_names: List[str] = None,
                            verbose: bool = True,
                            out_path: str = None) -> Tuple[Figure, Axes]:
        return self.explorer.render_map_wo_image(
            nusc, sample_token, camera_channel=camera_channel, alpha=alpha,
            patch_radius=patch_radius, min_polygon_area=min_polygon_area,
            render_behind_cam=render_behind_cam, render_outside_im=render_outside_im,
            layer_names=layer_names, verbose=verbose, out_path=out_path)

class PerlDiff_NuScenesMapExplorer(NuScenesMapExplorer):
    """ Helper class to explore the nuScenes map data. """
    def __init__(self,
                 map_api: NuScenesMap,
                 representative_layers: Tuple[str] = ('drivable_area', 'lane', 'walkway'),
                 color_map: dict = None):
        """
        :param map_api: NuScenesMap database class.
        :param representative_layers: These are the layers that we feel are representative of the whole mapping data.
        :param color_map: Color map.
        """
        # Mutable default argument.
        if color_map is None:
            color_map = dict(drivable_area='#a6cee3',
                             road_segment='#1f78b4',
                             road_block='#b2df8a',
                             lane='#33a02c',
                             ped_crossing='#fb9a99',
                             walkway='#e31a1c',
                             stop_line='#fdbf6f',
                             carpark_area='#ff7f00',
                             road_divider='#cab2d6',
                             lane_divider='#6a3d9a',
                             traffic_light='#7e772e')

        self.map_api = map_api
        self.representative_layers = representative_layers
        self.color_map = color_map

        self.canvas_max_x = self.map_api.canvas_edge[0]
        self.canvas_min_x = 0
        self.canvas_max_y = self.map_api.canvas_edge[1]
        self.canvas_min_y = 0
        self.canvas_aspect_ratio = (self.canvas_max_x - self.canvas_min_x) / (self.canvas_max_y - self.canvas_min_y)

    def render_map_wo_image(self,
                            nusc: NuScenes,
                            sample_token: str,
                            camera_channel: str = 'CAM_FRONT',
                            alpha: float = 0.3,
                            patch_radius: float = 10000,
                            min_polygon_area: float = 1000,
                            render_behind_cam: bool = True,
                            render_outside_im: bool = True,
                            layer_names: List[str] = None,
                            verbose: bool = True,
                            out_path: str = None) -> Tuple[Figure, Axes]:
        """
        Render a nuScenes camera image and overlay the polygons for the specified map layers.
        Note that the projections are not always accurate as the localization is in 2d.
        :param nusc: The NuScenes instance to load the image from.
        :param sample_token: The image's corresponding sample_token.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param alpha: The transparency value of the layers to render in [0, 1].
        :param patch_radius: The radius in meters around the ego car in which to select map records.
        :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
        :param render_behind_cam: Whether to render polygons where any point is behind the camera.
        :param render_outside_im: Whether to render polygons where any point is outside the image.
        :param layer_names: The names of the layers to render, e.g. ['lane'].
            If set to None, the recommended setting will be used.
        :param verbose: Whether to print to stdout.
        :param out_path: Optional path to save the rendered figure to disk.
        """
        near_plane = 1e-8

        if verbose:
            print('Warning: Note that the projections are not always accurate as the localization is in 2d.')

        # Default layers.
        if layer_names is None:
            layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']

        # Check layers whether we can render them.
        for layer_name in layer_names:
            assert layer_name in self.map_api.non_geometric_polygon_layers, \
                'Error: Can only render non-geometry polygons: %s' % layer_names

        # Check that NuScenesMap was loaded for the correct location.
        sample_record = nusc.get('sample', sample_token)
        scene_record = nusc.get('scene', sample_record['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']
        assert self.map_api.map_name == log_location, \
            'Error: NuScenesMap loaded for location %s, should be %s!' % (self.map_api.map_name, log_location)

        # Grab the front camera image and intrinsics.
        cam_token = sample_record['data'][camera_channel]
        cam_record = nusc.get('sample_data', cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)
        # im = Image.open(cam_path)
        ########################### modified ##########################
        original_im = Image.open(cam_path)
        width, height = original_im.size
        im = Image.new('RGB', (width, height), (255, 255, 255))
        im_size = im.size
        cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

        # Retrieve the current map.
        poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        box_coords = (
            ego_pose[0] - patch_radius,
            ego_pose[1] - patch_radius,
            ego_pose[0] + patch_radius,
            ego_pose[1] + patch_radius,
        )
        records_in_patch = self.get_records_in_patch(box_coords, layer_names, 'intersect')

        # Init axes.
        fig = plt.figure(figsize=(9, 16))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, im_size[0])
        ax.set_ylim(0, im_size[1])
        ax.imshow(im)

        # Retrieve and render each record.
        for layer_name in layer_names:
            for token in records_in_patch[layer_name]:
                record = self.map_api.get(layer_name, token)
                if layer_name == 'drivable_area':
                    polygon_tokens = record['polygon_tokens']
                else:
                    polygon_tokens = [record['polygon_token']]

                for polygon_token in polygon_tokens:
                    polygon = self.map_api.extract_polygon(polygon_token)

                    # Convert polygon nodes to pointcloud with 0 height.
                    points = np.array(polygon.exterior.xy)
                    points = np.vstack((points, np.zeros((1, points.shape[1]))))

                    # Transform into the ego vehicle frame for the timestamp of the image.
                    points = points - np.array(poserecord['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

                    # Transform into the camera.
                    points = points - np.array(cs_record['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)

                    # Remove points that are partially behind the camera.
                    depths = points[2, :]
                    behind = depths < near_plane
                    if np.all(behind):
                        continue

                    if render_behind_cam:
                        # Perform clipping on polygons that are partially behind the camera.
                        points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)
                    elif np.any(behind):
                        # Otherwise ignore any polygon that is partially behind the camera.
                        continue

                    # Ignore polygons with less than 3 points after clipping.
                    if len(points) == 0 or points.shape[1] < 3:
                        continue

                    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                    points = view_points(points, cam_intrinsic, normalize=True)

                    # Skip polygons where all points are outside the image.
                    # Leave a margin of 1 pixel for aesthetic reasons.
                    inside = np.ones(points.shape[1], dtype=bool)
                    inside = np.logical_and(inside, points[0, :] > 1)
                    inside = np.logical_and(inside, points[0, :] < im.size[0] - 1)
                    inside = np.logical_and(inside, points[1, :] > 1)
                    inside = np.logical_and(inside, points[1, :] < im.size[1] - 1)
                    if render_outside_im:
                        if np.all(np.logical_not(inside)):
                            continue
                    else:
                        if np.any(np.logical_not(inside)):
                            continue

                    points = points[:2, :]
                    points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                    polygon_proj = Polygon(points)

                    # Filter small polygons
                    if polygon_proj.area < min_polygon_area:
                        continue

                    label = layer_name
                    ax.add_patch(descartes.PolygonPatch(polygon_proj, fc=self.color_map[layer_name], alpha=alpha,
                                                        label=label))

        # Display the image.
        plt.axis('off')
        ax.invert_yaxis()

        if out_path is not None:
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

        return fig, ax    

class NuscDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        version,
        dataroot,
        is_train,
        data_aug_conf,
        max_boxes_per_sample=0,
        random_crop = False,
        random_flip = True,
    ):
        self.nusc = NuScenes(
            version="v1.0-{}".format(version),
            # dataroot=os.path.join(dataroot, version),
            dataroot=os.path.join(dataroot),
            verbose=False,
        )

        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.random_flip = random_flip
        self.pil_to_tensor = transforms.PILToTensor()
        self.max_boxes = max_boxes_per_sample
        self.scenes = self.get_scenes()
        self.samples = self.prepro()

    def total_images(self):
        return len(self)

    def get_scenes(self):
        # filter by scene split
        split = {
            "v1.0-trainval": {True: "train", False: "val"},
            # "v1.0-trainval": {True: "train", False: "val"},Actually, there are also test sets
            "v1.0-mini": {True: "mini_train", False: "mini_val"},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]
        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [
            samp
            for samp in samples
            if self.nusc.get("scene", samp["scene_token"])["name"] in self.scenes
        ]

        # sort by scene, timestamp (only to make chronological viz easier)
        # samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        return samples

    def get_camera_info(self, sample_record, cams, index=None):
        for cam in cams:
            sample_data_token = sample_record["data"][cam]
            # Retrieve sensor & pose records
            sd_record = self.nusc.get("sample_data", sample_data_token)
            if not sd_record["is_key_frame"]:
                raise ValueError("not keyframes")

            s_record = self.nusc.get("sample", sd_record["sample_token"])
            scene_token = sample_record["scene_token"]
            scene = self.nusc.get("scene", scene_token)
            log_token = scene['log_token']
            log = self.nusc.get("log", log_token)
            map_name = log['location']

            nusc_map = PerlDiff_NuscenesMap(
                dataroot=self.nusc.dataroot,
                map_name=map_name,
            )
            layer_names = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                                             'walkway', 'stop_line', 'carpark_area']
            img_name = os.path.join(self.nusc.dataroot, sd_record["filename"])
            out_path = img_name.replace('samples', 'samples_road_map_test')
            par_out_path = '/'.join(out_path.split('/')[:-1])
            if not os.path.exists(par_out_path):
                os.makedirs(par_out_path)
            nusc_map.render_map_wo_image(self.nusc, s_record['token'], layer_names=layer_names, camera_channel=sd_record['channel'], out_path=out_path, verbose=False)
        
        return  sample_record

    def choose_cams(self):
        if self.data_aug_conf["Ncams"] < len(
            self.data_aug_conf["cams"]
        ):
            cams = [self.data_aug_conf["cams"][1]]
        else:
            cams = self.data_aug_conf["cams"]
        return cams
    
    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_data_record = self.samples[index]
        
        cams = self.choose_cams()

        _ = self.get_camera_info(sample_data_record, cams)
    

if __name__ == "__main__":
    version = "trainval"
    dataroot = "/disk1/jinhua.zjh/DATA/nuscenes"
    final_dim = (256, 384)

    data_aug_conf = {
        "final_dim": final_dim,
        "cams": [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
        ],
        "Ncams": 6,
    }
    nusc = NuScenes(
        version="v1.0-{}".format(version),
        # dataroot=os.path.join(dataroot, version),
        dataroot=os.path.join(dataroot),
        verbose=False,
    )
    traindata = NuscDataset(version, dataroot, is_train=True,
                         data_aug_conf=data_aug_conf, max_boxes_per_sample=80)
    

    valdata = NuscDataset(version, dataroot, is_train=False,
                         data_aug_conf=data_aug_conf, max_boxes_per_sample=80)

    

    for i in tqdm(range(len(traindata))):
        _ = traindata[i]

    for j in tqdm(range(len(valdata))):
        
        _ = valdata[i]

