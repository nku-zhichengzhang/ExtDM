from functools import partial
import torch.nn as nn
import torch
from model.newDM.new_video_flow_diffusion import ResnetBlock, MotionAdaptor

class RefineModule(nn.Module):
    def __init__(self, cond_num, pred_num, channels=3, middle_dim=8, adaptor_dim=64):
        super(RefineModule, self).__init__()
        self.frame_num = cond_num + pred_num
        
        self.s_conv  = nn.Conv3d(channels, middle_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.t_conv  = nn.Conv3d(middle_dim, middle_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.adaptor = MotionAdaptor(middle_dim*self.frame_num, adaptor_dim, middle_dim*pred_num)
        self.final_conv = nn.Conv3d(middle_dim, channels, 1)
    
    def forward(self, cond, pred):
        # b c t h w
        video = torch.cat([cond, pred], dim=2)
        print(video.shape)
        video = self.s_conv(video)
        print(video.shape)
        video = self.t_conv(video)
        print(video.shape)
        video = self.adaptor(video)
        print(video.shape)
        video = self.final_conv(video)
        print(video.shape)
        return video
    
# cond = torch.zeros((8,3,10,64,64))
# pred = torch.zeros((8,3,20,64,64))

# refine = RefineModule(
#     cond_num=10, 
#     pred_num=20, 
#     channels=3, 
#     middle_dim=8, 
#     adaptor_dim=64
# )

# res = refine(cond, pred)
# print(res.shape)