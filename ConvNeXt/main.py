from convNeXt import convnext_large as convnext
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import torchvision
from tqdm import tqdm
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from convNeXt import TrainingConfig, Trainer

DATASET_PATH = "./data/"

transform=transforms.Compose([ 
                        transforms.RandomCrop((32,32), padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                std=(0.2675, 0.2565, 0.2761))
                ])

train_dataset = CIFAR100(root=DATASET_PATH, train=True, download=True, transform=transform)
test_dataset = CIFAR100(root=DATASET_PATH, train=False, download=True, transform=transform)

model = convnext(in_channels=3, num_classes=100, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.)

trainig_config = TrainingConfig(max_epochs=100, batch_size=128, learning_rate=4e-3, weight_decay = 0.05)
trainer = Trainer(model=model, train_dataset=train_dataset, test_dataset=test_dataset, configs=trainig_config)
trainer.train()

print(model)
print(model.parameters())
