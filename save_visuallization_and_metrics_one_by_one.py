import os
import torch
from tqdm import tqdm
from utils.visualize import visualize

from metrics.calculate_fvd import calculate_fvd,calculate_fvd1
from metrics.calculate_psnr import calculate_psnr,calculate_psnr1
from metrics.calculate_ssim import calculate_ssim,calculate_ssim1
from metrics.calculate_lpips import calculate_lpips,calculate_lpips1
import json

# test_name='./logs_validation/diffusion/snapshots-joint-steplr-random-onlyflow-train-regionmm-temp'
# test_name='./logs_validation/diffusion/kth64_DM_Batch32_lr2e-4_c10p5_0825_fix'
# test_name='./logs_validation/diffusion/kth64_DM_Batch32_lr2e-4_c10p5_0825_random'

# test_name='./logs_validation/diffusion/bair64_DM_Batch32_lr2e-4_c2p7'

# test_name='./logs_validation/flow/kth64_FlowAE_Batch256_lr2e-4'

test_name='./logs_validation/flow/carla128_FlowAE_Batch128_lr1e-4_Region20_perspective'

num_frames_cond = 10
# num_frames_cond = 2

# video_num = 256
video_num = 100

# ps: pixel value for metrics should be in [0, 1]!

device = torch.device("cuda")

metrics = {}

videos1 = torch.load(f'{test_name}/origin.pt')
videos2 = torch.load(f'{test_name}/result.pt')

def metrics_by_video(videos1, videos2):
    with open(f'{test_name}/metrics.csv', "w") as f:
            f.write('id,psnr,ssim,lpips,fvd\n')
            
    for i in tqdm(range(video_num)):
        video1 = videos1[i:i+1]
        video2 = videos2[i:i+1]
        metrics['fvd'] = calculate_fvd1(video1, video2, device)
        # video1 = video1[:,num_frames_cond:]
        # video2 = video2[:,num_frames_cond:]
        metrics['ssim'] = calculate_ssim1(video1, video2)[0]
        metrics['psnr'] = calculate_psnr1(video1, video2)[0]
        metrics['lpips'] = calculate_lpips1(video1, video2, device)[0]
        
        with open(f'{test_name}/metrics.csv', "a") as f:
            f.write(f"{i},{metrics['psnr']:.6},{metrics['ssim']:.6},{metrics['lpips']:.6},{metrics['fvd']:.6}\n")

# 按帧算
def metrics_by_frame(videos1, videos2):

    print(videos1.shape, torch.min(videos1), torch.max(videos1), torch.mean(videos1), torch.std(videos1))
    print(videos2.shape, torch.min(videos2), torch.max(videos2), torch.mean(videos2), torch.std(videos2))
    for i in range(2, 18, 2):
        metrics['fvd'] = calculate_fvd1(videos1, videos2, device, mini_bs=i)
        print(metrics)

    metrics['fvd'] = calculate_fvd(videos1, videos2, device, mini_bs=2)
    videos1 = videos1[:,num_frames_cond:]
    videos2 = videos2[:,num_frames_cond:]
    metrics['ssim'] = calculate_ssim(videos1, videos2)
    metrics['psnr'] = calculate_psnr(videos1, videos2)
    metrics['lpips'] = calculate_lpips(videos1, videos2, device)

    for metrics_name in ['fvd', 'ssim', 'psnr', 'lpips']:
        with open(f'{test_name}/framewise-{metrics_name}.csv', "w") as f:
            f.write(f'frame,value\n')
            for timestamp, value in metrics[metrics_name][metrics_name].items():
                t = int(timestamp.split('[')[-1].split(']')[0]) + (num_frames_cond if metrics_name != 'fvd' else 0)
                f.write(f'{t},{value}\n')

def show_videos(videos1, videos2):        
    visualize(
        save_path=test_name,
        origin=videos1,
        result=videos2,
        save_pic_num=10,
        select_method='top',
        grid_nrow=5,
        save_gif_grid=True,
        save_gif=False,
        save_pic_row=True,
        save_pic=True,
        skip_pic_num=5,
        cond_frame_num=num_frames_cond,   
    )

# metrics_by_video(videos1, videos2)
metrics_by_frame(videos1, videos2)
# show_videos(videos1, videos2)