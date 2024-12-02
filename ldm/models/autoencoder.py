import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 scale_factor=1
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor

    def encode(self, x):
        B, N, _, _, _ = x.shape
        x = rearrange(x, 'b n c h w -> (b n) c h w')
        h = self.encoder(x) #b*(2*z_channels)*h*w (downsample with ch_mult)
        moments = self.quant_conv(h) # b*(2*embed_dim)*h*w
        posterior = DiagonalGaussianDistribution(moments) # bs*embed_dim*h*w
        posterior = posterior.sample() * self.scale_factor
        posterior = rearrange(posterior, '(b n) c h w -> b n c h w', b=B, n=N)
        return posterior

    def decode(self, z):
        z = 1. / self.scale_factor * z # # bs*embed_dim*h*w
        B, N, _, _, _ = z.shape
        z = rearrange(z, 'b n c h w -> (b n) c h w')
        z = self.post_quant_conv(z) # 1*z_channels*h*w*
        dec = self.decoder(z) # size is similiar with x
        dec = rearrange(dec, '(b n) c h w -> b n c h w', b=B, n=N)
        return dec

if __name__ == "__main__":
    ddconfig = {
    'double_z': True,
    'z_channels': 4,
    'resolution': 512,
    'in_channels': 3,
    'out_ch': 3,
    "ch": 128,
    "ch_mult":[1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0
    }

    autoencoder = AutoencoderKL(ddconfig, embed_dim=40)
    input_image = torch.randn(1, 6, 3, 32, 32) 
    encoder_output = autoencoder.encode(input_image) # bs,embed_dim, down_h, dow_w
    print(encoder_output.shape)
    input_image = torch.randn(1, 6, 40, 4, 4) # bs, embed_dim,
    decoder_output = autoencoder.decode(input_image)
    print(decoder_output.shape)







