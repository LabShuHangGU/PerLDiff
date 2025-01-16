
import os 

class DatasetCatalog:
    def __init__(self, ROOT):   
       
       
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #    
        self.Kitti = {   
            "target": "dataset.kitti_dataset.KittiDataset",
            "train_params":dict(
                data_aug_conf = {
                    'final_dim': (384, 1280),
                    'cams': ['CAM_LEFT', 'CAM_RIGHT'],
                    'Ncams': 1,
                },
                # version = "trainval",
                dataroot = os.path.join(ROOT, 'kitti'),
                is_train = True
            ),
            "val_params":dict(
                data_aug_conf = {
                    'final_dim': (384, 1280),
                    'cams': ['CAM_LEFT', 'CAM_RIGHT'],
                    'Ncams': 1,
                },
                # version = "trainval",
                dataroot = os.path.join(ROOT, 'kitti'),
                is_train = False
            )
        }
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #    
        self.Kitti_with_path = {   
            "target": "dataset.kitti_dataset_with_path.KittiDataset",
            "train_params":dict(
                data_aug_conf = {
                    'final_dim': (384, 1280),
                    'cams': ['CAM_LEFT', 'CAM_RIGHT'],
                    'Ncams': 1,
                },
                # version = "trainval",
                dataroot = os.path.join(ROOT, 'kitti'),
                is_train = True
            ),
            "val_params":dict(
                data_aug_conf = {
                    'final_dim': (384, 1280),
                    'cams': ['CAM_LEFT', 'CAM_RIGHT'],
                    'Ncams': 1,
                },
                # version = "trainval",
                dataroot = os.path.join(ROOT, 'kitti'),
                is_train = False
            )
        }


