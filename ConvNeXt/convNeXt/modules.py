import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class ConvNeXtBlock(nn.Module):
    """ The architecture of this block is as follows :
    
    DepthWise conv -> Permute to (N, H, W, C); [Channel Last]; Layer_norm -> Linear -> GELU -> Linear -> Permute Back

    Channel Last is used in input dimensions because its faster in PyTorch
    
    """

    def __init__(self, in_channel , depth_rate=0., layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()

        """Using Group covolution using groups as in the in_channel so it behaves as Depth Wise Convolution"""
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel)

        self.norm = Layer_norm(in_channel, eps=1e-6)
        
        """point wise convolution with 1x1 conv is similar to a Linear Layer"""
        self.pwconv1 = nn.Linear(in_channel, 4*in_channel)

        self.activation = nn.GELU()

        self.pwconv2 = nn.Linear(4*in_channel, in_channel)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channel)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        """Stochastic Depth aims to shrink the depth of a network during training, 
        while keeping it unchanged during testing. This is achieved by randomly dropping 
        entire ResBlocks during training and bypassing their transformations through 
        skip connections."""
        self.dropPath = DropPath(depth_rate) if depth_rate > 0. else nn.Identity()


    def forward(self,x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.dropPath(x)

        return x

class Layer_norm(nn.Module):

    def __init__(self, normShape, eps=1e-6, input_format="Channel_Last"):
        super(Layer_norm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normShape))
        self.bias = nn.Parameter(torch.zeros(normShape))
        self.eps = eps
        self.dataFormat = input_format
        if self.dataFormat not in ["Channel_Last", "Channel_First"]:
            raise NotImplementedError
        self.normShape = (normShape, )

    def forward(self, x):
        if self.dataFormat == "Channel_Last":
            return F.layer_norm(x, self.normShape, self.weight, self.bias, self.eps)
        elif self.dataFormat == "Channel_First":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

            