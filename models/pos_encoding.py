import math

import torch
from torch import nn
from boxx import *

class PositionEncodingSine1DRelative(nn.Module):
    """
    relative sine encoding 1D, partially inspired by DETR (https://github.com/facebookresearch/detr)
    """

    def __init__(self, num_pos_feats=1, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @torch.no_grad()
    def forward(self, inputs):
        """
        :param inputs: NestedTensor
        :return: pos encoding [N,C,H,2W-1]
        """
        x = inputs
        tree-x
        # update h and w if downsampling
        bs, _, h, w = x.size()

        # populate all possible relative distances
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=x.device)

        # scale distance if there is down sample
        # if inputs.sampled_cols is not None:
        #     scale = x.size(-1) / float(inputs.sampled_cols.size(-1))
        #     x_embed = x_embed * scale

        if self.normalize:
            x_embed = x_embed * self.scale

        # dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # tree-dim_t
        tree-x_embed
        pos_x = x_embed[:, None]   # 2W-1xC
        # interleave cos and sin instead of concatenate
        pos = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC

        return pos


def forward_single(x):                                                                                                                       
    """                                                                                                                                       
    CoordConv from solo                                                                                                                       
    """                                                                                                                                       
    ins_kernel_feat = x                                                                                                                       
    # ins branch                                                                                                                              
    # concat coord                                                                                                                            
    x_range = torch.linspace(                                                                                                                 
        1,                                                                                                                                    
        ins_kernel_feat.shape[-1],                                                                                                            
        ins_kernel_feat.shape[-1],                                                                                                            
        device=ins_kernel_feat.device,                                                                                                        
    )                                                                                                                                         
    y_range = torch.linspace(                                                                                                                 
        1,                                                                                                                                    
        ins_kernel_feat.shape[-2],                                                                                                            
        ins_kernel_feat.shape[-2],                                                                                                            
        device=ins_kernel_feat.device,                                                                                                        
    )                                                                                                                                         
    # tree-x_range                                                                                                                            
    y, x = torch.meshgrid(y_range, x_range)                                                                                                   
    y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])                                                                                       
    x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])                                                                                       
    coord_feat = torch.cat([x.sin(), y.cos()], 1)                                                                                             
    # tree-coord_feat                                                                                                                         
    ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)                                                                             
                                                                                                                                            
    return ins_kernel_feat  


if __name__ == "__main__":
    pos = PositionEncodingSine1DRelative()
    x = torch.rand((1,3,385,576))

    y = pos(x)
    tree-y