import torch.nn as nn
import torch
from model.newDM.new_video_flow_diffusion import MotionAdaptor

class RefineModule(nn.Module):
    def __init__(self, cond_num, pred_num, supervised=True, channels=3, middle_dim=8, adaptor_dim=64):
        super(RefineModule, self).__init__()
        
        self.supervised = supervised
        
        if self.supervised:
            self.frame_num = cond_num + pred_num
        else:
            self.frame_num = pred_num
        
        self.s_conv  = nn.Conv3d(channels, middle_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.t_conv  = nn.Conv3d(middle_dim, middle_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        
        self.adaptor = MotionAdaptor(middle_dim*self.frame_num, adaptor_dim, middle_dim*pred_num)
        self.final_conv = nn.Conv3d(middle_dim, channels, 1)
        
    
    def forward(self, pred, cond=None):
        
        if self.supervised:
            assert cond is not None, "cond cannnot be None"
            video = torch.cat([cond, pred], dim=2)
        else:
            assert cond is None, "cond must be None"
            video = pred
        
        video = self.s_conv(video)
        video = self.t_conv(video)
        video = self.adaptor(video)
        video = self.final_conv(video)
        return video
    
# cond = torch.zeros((8,3,10,64,64))
# pred = torch.zeros((8,3,20,64,64))

# refine = RefineModule(
#     cond_num=10, 
#     pred_num=20, 
#     channels=3, 
#     supervised=True,
#     middle_dim=8, 
#     adaptor_dim=64
# )

# refine2 = RefineModule(
#     cond_num=10, 
#     pred_num=20, 
#     channels=3, 
#     supervised=False,
#     middle_dim=8, 
#     adaptor_dim=64
# )

# res = refine(pred=pred, cond=cond)
# print(res.shape)

# res2 = refine2(pred=pred)
# print(res2.shape)