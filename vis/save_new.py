import os
import torch
from tqdm import tqdm
from utils.visualize import visualize

from metrics.calculate_fvd import calculate_fvd,calculate_fvd1
from metrics.calculate_psnr import calculate_psnr,calculate_psnr1
from metrics.calculate_ssim import calculate_ssim,calculate_ssim1
from metrics.calculate_lpips import calculate_lpips,calculate_lpips1
import json

test_name='/home/ubuntu/zzc/code/EDM/logs_validation/diffusion/cityscapes128_DM_Batch44_lr1.5e-4_c2p4_STW_adaptor'

num_frames_cond = 2
video_num = 256

device = torch.device("cuda")

metrics = {}

videos1 = torch.load(f'{test_name}/origin.pt')
videos2 = torch.load(f'{test_name}/result.pt')
flows1 = torch.load(f'{test_name}/origin_flows.pt')
flows2 = torch.load(f'{test_name}/result_flows.pt')

abs_diff_videos = torch.sqrt(torch.sum((videos1 - videos2)**2, dim=2)/3).unsqueeze(2).repeat(1,1,3,1,1)
abs_diff_flows  = torch.sqrt(torch.sum((flows1 - flows2)**2, dim=2)/3).unsqueeze(2).repeat(1,1,3,1,1)

from utils.visualize import visualize_ori_pre_flow_diff
visualize_ori_pre_flow_diff(
    save_path=f"{test_name}/result",
    origin=videos1, 
    result=videos2, 
    origin_flow=flows1, 
    result_flow=flows2, 
    video_diff=abs_diff_videos, 
    flow_diff=abs_diff_flows, 
    epoch_or_step_num=0, 
    cond_frame_num=num_frames_cond, 
    skip_pic_num=1
)

from metrics.calculate_fvd import calculate_fvd,calculate_fvd1
from metrics.calculate_psnr import calculate_psnr,calculate_psnr1
from metrics.calculate_ssim import calculate_ssim,calculate_ssim1
from metrics.calculate_lpips import calculate_lpips,calculate_lpips1

fvd = calculate_fvd1(videos1, videos2, torch.device("cuda"), mini_bs=16)
videos1 = videos1[:, num_frames_cond:]
videos2 = videos2[:, num_frames_cond:]
ssim = calculate_ssim1(videos1, videos2)[0]
psnr = calculate_psnr1(videos1, videos2)[0]
lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]

print("[fvd    ]", fvd)
print("[ssim   ]", ssim)
print("[psnr   ]", psnr)
print("[lpips  ]", lpips)