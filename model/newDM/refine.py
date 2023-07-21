from functools import partial
import torch.nn as nn
import torch
from model.newDM.new_video_flow_diffusion import ResnetBlock, MotionAdaptor, PreNorm, Residual, SpatialLinearAttention, temporal_attn

class RefineModule(nn.Module):
    def __init__(self, cond_num, pred_num, channels=3, middle_dim=64):
        super(RefineModule, self).__init__()
        self.frame_num = cond_num + pred_num
        self.adaptor = MotionAdaptor(channels*self.frame_num, middle_dim, channels*self.frame_num)
        
        block_klass = partial(ResnetBlock, groups=8)
        
        self.final_conv = nn.Sequential(
            block_klass(channels * 2, channels),
            nn.Conv3d(channels, channels, 1)
        )
    
    def forward(self, cond, pred):
        # b c t h w
        video = torch.cat([cond, pred], dim=2)
        video = self.adaptor(video)
        video = self.final_conv(video)
        return video