import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F
from einops import rearrange, repeat
    
class EncodingNet(nn.Module):
    def __init__(self,  in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*16 # 2 is sin&cos, 16 is 8*[x,y] 

        self.linears = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, box, box_mask, positive_embeddings):
        B, num_camera, N, _ = box.shape 
        box_mask = box_mask.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(box) # B*N*16 --> B*N*C

        # learnable null embedding 
        positive_null = self.null_positive_feature.view(1,1,1,-1)
        xyxy_null =  self.null_position_feature.view(1,1,1,-1)

        # replace padding with learnable null embedding 
        positive_embeddings = positive_embeddings * box_mask + (1- box_mask) * positive_null
        xyxy_embedding = xyxy_embedding * box_mask + (1- box_mask) * xyxy_null

        objs = self.linears(  torch.cat([positive_embeddings, xyxy_embedding], dim=-1)  )
        assert objs.shape == torch.Size([B, num_camera, N, self.out_dim])        
        return objs