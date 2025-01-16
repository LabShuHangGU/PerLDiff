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

fid_value = fid.compute_fid(path1, path2)


print(f"Average FID: {fid_value}")
