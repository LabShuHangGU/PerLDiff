
import os 

class DatasetCatalog:
    def __init__(self, ROOT):   
       
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #    
        self.Nuscenes = {   
            "target": "dataset.nuscenes_dataset.NuscDataset",
            "train_params":dict(
                data_aug_conf = {
                    'final_dim': (256, 384),
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'],
                    'Ncams': 6,
                },
                version = "trainval",
                dataroot = os.path.join(ROOT, 'nuscenes'),
                is_train = True
            ),
            "val_params":dict(
                data_aug_conf = {
                    'final_dim': (256, 384),
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'],
                    'Ncams': 6,
                },
                version = "trainval",
                dataroot = os.path.join(ROOT, 'nuscenes'),
                is_train = False
            )
        }
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #    
        self.Nuscenes_with_path = {   
            "target": "dataset.nuscenes_dataset_with_path.NuscDataset",
            "train_params":dict(
                data_aug_conf = {
                    'final_dim': (256, 384),
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'],
                    'Ncams': 6,
                },
                version = "trainval",
                dataroot = os.path.join(ROOT, 'nuscenes'),
                is_train = True
            ),
            "val_params":dict(
                data_aug_conf = {
                    'final_dim': (256, 384),
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'],
                    'Ncams': 6,
                },
                version = "trainval",
                dataroot = os.path.join(ROOT, 'nuscenes'),
                is_train = False
            )
        }
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #    

