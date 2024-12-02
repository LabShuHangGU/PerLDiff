import os
import matplotlib.pyplot as plt

output_folder = "/disk1/jinhua.zjh/DATA/nuscenes/samples_road_map_test"

os.makedirs(output_folder, exist_ok=True)

cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


for cam in cams:
    search_path = os.path.join(output_folder, cam)
    print(f"svae path is {search_path}")

    png_files = [f for f in os.listdir(search_path) if f.endswith('.jpg') or f.endswith('.png')]

    n = len(png_files)
    print(f"the sum of jpg file in {cam} is {n}")
