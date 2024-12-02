import numpy as np
import os
import argparse
from cleanfid import fid

parser = argparse.ArgumentParser(description='Calculate FID between two sets of images.')
parser.add_argument('path1', type=str, help='Path to the first set of images.')
parser.add_argument('path2', type=str, help='Path to the second set of images.')


args = parser.parse_args()
path1 = args.path1
path2 = args.path2

cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

sum_fid = []
for cam in cams:
    search_path1 = os.path.join(path1, cam)
    search_path2 = os.path.join(path2, cam)

    fid_value = fid.compute_fid(search_path1, search_path2)

    sum_fid.append(fid_value)
    print(f"{cam} FID: {fid_value}")

average_fid = np.mean(sum_fid)
print(f"Average FID: {average_fid}")
