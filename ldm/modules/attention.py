import os
import math
from inspect import isfunction

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.utils import checkpoint
import torchvision
from torchvision.utils import make_grid

from einops import rearrange, repeat

from ldm import util


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)



class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0, atten_map_res=[32, 48], max_boxes=80, max_length=77):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads


        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)


        self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )


        self.atten_map_res = atten_map_res
        self.max_boxes = max_boxes
        self.max_length = max_length

        self.threshold = -0.001

        self.lamda1 = 5.0
        self.lamda2 = 5.0      


    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B,M = mask.shape
            mask = mask.unsqueeze(1).repeat(1,self.heads,1).reshape(B*self.heads,1,-1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim 


    def forward(self, x, key, value, mask=None, perl_box_masking_map=None, perl_road_masking_map=None):

        q = self.to_q(x)     # B*N*(H*C)
        k = self.to_k(key)   # B*M*(H*C)
        v = self.to_v(value) # B*M*(H*C)
   
        B, N, HC = q.shape 
        _, M, _ = key.shape
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
        v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale # (B*H)*N*M
        self.fill_inf_from_mask(sim, mask)

        if perl_box_masking_map != None:
            assert M == perl_box_masking_map.shape[-1] # B*N*M
            perl_box_masking_map = repeat(perl_box_masking_map, 'b n m->(b h) n m', h=H) # (B*H)*N*M

            sim = sim + self.lamda1 * perl_box_masking_map # (B*H)*N*M

        if perl_road_masking_map != None:
            perl_road_masking_map = repeat(perl_road_masking_map, "b m n->(b h) n m", h=H,m=1) # B*1*N -> (B*H)*N*M

            sim = sim + self.lamda2 * perl_road_masking_map # (B*H)*N*M

        attn = sim.softmax(dim=-1) # (B*H)*N*M

        out = torch.einsum('b i j, b j d -> b i d', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)
    

class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)


class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()
        
        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head) 
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, objs, perl_box_masking_map=None, perl_road_masking_map=None):
        
        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn( self.norm1(x), objs, objs, perl_box_masking_map=perl_box_masking_map, perl_road_masking_map=perl_road_masking_map) 
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) ) 
        
        return x 



class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True, num_camera=6):
        super().__init__()
        self.num_camera= num_camera
        self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)  
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head) 

        self.cross_view_left = CrossAttention(query_dim=query_dim, key_dim=query_dim, value_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.cross_view_right = CrossAttention(query_dim=query_dim, key_dim=query_dim, value_dim=query_dim, heads=n_heads, dim_head=d_head)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

        self.use_checkpoint = use_checkpoint

        if fuser_type == "gatedCA":
            self.fuser = GatedCrossAttentionDense(query_dim, key_dim, value_dim, n_heads, d_head) 
            self.attn_back = GatedCrossAttentionDense(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, n_heads=n_heads, d_head=d_head)
        else:
            assert False, f"fuser_type={fuser_type} is not supported!!!"


    def forward(self, x, context, objs, perl_box_masking_map, perl_road_masking_map, road_map_embedding):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs, perl_box_masking_map, perl_road_masking_map, road_map_embedding)
        else:
            return self._forward(x, context, objs, perl_box_masking_map, perl_road_masking_map, road_map_embedding)

    def _forward(self, x, context, objs, perl_box_masking_map, perl_road_masking_map, road_map_embedding): 
        x = self.attn1( self.norm1(x) ) + x 
        
        if road_map_embedding != None:
            x = self.attn_back(x, road_map_embedding, perl_road_masking_map=perl_road_masking_map) + x
        
        x = self.fuser(x, objs, perl_box_masking_map=perl_box_masking_map) # identity mapping in the beginning 

        BN, L, C = x.shape
        B = BN // self.num_camera
        x = rearrange(x, "(b n) l c-> b n l c",b=B, n=self.num_camera) # [B, N, L, C]

        x_left = torch.roll(x, 1, 1)
        x_right = torch.roll(x, -1, 1)

        x = rearrange(x, "b n l c-> (b n) l c",b=B, n=self.num_camera) # [B*N, L, C]
        x_left = rearrange(x_left, "b n l c-> (b n) l c",b=B, n=self.num_camera) # [B*N, L, C]
        x_right = rearrange(x_right, "b n l c-> (b n) l c",b=B, n=self.num_camera) # [B*N, L, C]

        x = self.cross_view_left(x, x_left, x_left) + self.cross_view_right(x, x_right, x_right) + x
        
        x = self.attn2(self.norm2(x), context, context) + x

        x = self.ff(self.norm3(x)) + x

        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, fuser_type=None, use_checkpoint=True, num_camera=6):
        super().__init__()
        self.in_channels = in_channels
        query_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.num_camera = num_camera

        
        self.proj_in = nn.Conv2d(in_channels,
                                 query_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=use_checkpoint, num_camera=num_camera)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context, objs, perl_box_masking_map=None, perl_road_masking_map=None, road_map_embedding=None):
        n =  self.num_camera

        bn, c, h, w = x.shape
        b = bn // n

        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)

        x = rearrange(x, '(b n) c h w -> (b n) (h w) c',b=b,n=n,c=c,h=h,w=w)

        if perl_box_masking_map != None:     
            _, m, _, _= perl_box_masking_map.shape # (b n) m h w

            perl_box_masking_map = rearrange(perl_box_masking_map, "(b n) m h w->(b n m) h w", b=b, n=n,m=m)
            perl_box_masking_map = perl_box_masking_map[:, None, :, :] # (b n m) 1 h w
            perl_box_masking_map = F.interpolate(perl_box_masking_map, size=(h, w), mode='bilinear', align_corners=False)

            perl_box_masking_map = perl_box_masking_map.reshape(-1, h, w) # (b n m) h w
            perl_box_masking_map = rearrange(perl_box_masking_map, '(b n m) h w-> (b n) (h w) m', m=m, b=b,n=n,h=h,w=w)

        if perl_road_masking_map != None:     
            perl_road_masking_map = perl_road_masking_map[:, None, :, :] # (b n) 1 h w
            perl_road_masking_map = F.interpolate(perl_road_masking_map, size=(h, w), mode='bilinear', align_corners=False)
            perl_road_masking_map = rearrange(perl_road_masking_map, '(b n) c h w-> (b n) c (h w)', b=b,n=n,h=h,w=w,c=1)

        for block in self.transformer_blocks:
            x = block(x, context, objs, perl_box_masking_map, perl_road_masking_map, road_map_embedding)

        x = rearrange(x, '(b n) (h w) c -> (b n) c h w', h=h, w=w,b=b,n=n)
        x = self.proj_out(x)
        return x + x_in