import argparse

import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
import random
import json_tricks as json
import yaml

from einops import rearrange

from torch.utils.data import DataLoader

from data.video_dataset import VideoDataset
from model.LFAE.flow_autoenc import FlowAE
from utils.meter import AverageMeter
from utils.seed import setup_seed

from model.DM.video_flow_diffusion_model_pred_condframe_temp import FlowDiffusion

if __name__ == "__main__":
    start = timeit.default_timer()
    cudnn.enabled = True
    cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description="Diffusion")
    parser.add_argument("--random-seed", 
                        type=int, 
                        default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--flowae_checkpoint",
                        default="./logs_training/flow/kth64_test/snapshots/RegionMM.pth")
    parser.add_argument("--checkpoint", 
                        default="/mnt/sda/hjy/training_logs/cityscapes128/snapshots/RegionMM.pth")
    parser.add_argument("--config", 
                        default="./config/carla128.yaml")
    parser.add_argument("--log_dir", 
                        default="./logs_validation/flow/flowautoenc_video_carla")
    parser.add_argument("--device_ids", 
                        default="0",
                        help="choose gpu device.")
    parser.add_argument("--fp16", 
                        default=False)
    args = parser.parse_args()

    setup_seed(args.random_seed)

    MEAN = (0.0, 0.0, 0.0)
    
    ckpt_dir = os.path.join(args.log_dir, "flowae_result")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model = FlowDiffusion(
        config=config,
        pretrained_pth=args.flowae_checkpoint,
        is_train=False,
    )
    model.cuda()

    checkpoint = torch.load(args.checkpoint)
    model.diffusion.load_state_dict(checkpoint['diffusion'])

    model.eval()

    dataset_params = config['dataset_params']
    train_params = config['diffusion_params']['train_params']
    cond_frames = dataset_params['valid_params']['cond_frames']
    total_pred_frames = dataset_params['valid_params']['pred_frames']
    pred_frames = dataset_params['train_params']['pred_frames']

    json_path = os.path.join(ckpt_dir, "loss%d.json" % (dataset_params['valid_params']['total_videos']))
        
    valid_dataset = VideoDataset(
        data_dir=dataset_params['root_dir'],
        type=dataset_params['valid_params']['type'], 
        image_size=dataset_params['frame_shape'],
        num_frames=dataset_params['valid_params']['cond_frames'] + dataset_params['valid_params']['pred_frames'], 
        mean=(0.0, 0.0, 0.0),
        total_videos=dataset_params['valid_params']['total_videos'],
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=train_params['valid_batch_size'],
        shuffle=False, 
        num_workers=train_params['dataloader_workers'],
        pin_memory=True, 
        drop_last=False
    )
    
    from math import ceil
    NUM_ITER    = ceil(dataset_params['valid_params']['total_videos'] / train_params['valid_batch_size'])
    NUM_AUTOREG = ceil(total_pred_frames / pred_frames)
    
    actual_step = int(ceil(checkpoint['example'] / train_params['batch_size']))
    
    # b t c h w [0-1]
    origin_videos = []
    result_videos = []

    for i_iter, batch in enumerate(valid_dataloader):
        if i_iter >= NUM_ITER:
            break

        real_vids, real_names = batch
        real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')

        origin_videos.append(real_vids)
        pred_video = []

        i_real_vids = real_vids[:,:,:cond_frames]

        for i_autoreg in range(NUM_AUTOREG):

            i_pred_video = model.sample_one_video(cond_scale=1.0, real_vid=i_real_vids.cuda())['sample_out_vid'][:,:,cond_frames:].clone().detach().cpu()
            print(f'[{i_autoreg}/{NUM_AUTOREG}] i_pred_video: {i_pred_video.shape}')

            pred_video.append(i_pred_video)

            i_real_vids = i_pred_video[:,:,-cond_frames:]  # get last cond frames (-cond_frames)

        pred_video = torch.cat(pred_video, dim=2)

        res_video = torch.cat([real_vids[:, :, :cond_frames], pred_video[:, :, :total_pred_frames]], dim=2)
        result_videos.append(res_video)

    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)

    origin_videos = rearrange(origin_videos, 'b c t h w -> b t c h w')
    result_videos = rearrange(result_videos, 'b c t h w -> b t c h w')
    
    from utils.visualize import visualize
    visualize(
        save_path=f"{args.log_dir}/video_result",
        origin=origin_videos,
        result=result_videos,
        save_pic_num=16,
        grid_nrow=4,
        save_gif_grid=True,
        save_pic_row=True,
        save_gif=False,
        epoch_or_step_num=actual_step, 
        cond_frame_num=cond_frames,
    )

    from metrics.calculate_fvd import calculate_fvd,calculate_fvd1
    from metrics.calculate_psnr import calculate_psnr,calculate_psnr1
    from metrics.calculate_ssim import calculate_ssim,calculate_ssim1
    from metrics.calculate_lpips import calculate_lpips,calculate_lpips1
    
    # fvd = calculate_fvd1(origin_videos, result_videos, torch.device("cuda"), mini_bs=2)
    # videos1 = origin_videos[:, cond_frames:cond_frames + pred_frames]
    # videos2 = result_videos[:, cond_frames:cond_frames + pred_frames]
    # ssim = calculate_ssim1(videos1, videos2)[0]
    # psnr = calculate_psnr1(videos1, videos2)[0]
    # lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
    
    # print("[fvd    ]", fvd)
    # print("[ssim   ]", ssim)
    # print("[psnr   ]", psnr)
    # print("[lpips  ]", lpips)
    
    # res_dict = {
    #     "fvd": fvd,
    #     "ssim": ssim,
    #     "psnr": psnr,
    #     "lpips": lpips
    # }
    
    videos1 = origin_videos
    videos2 = result_videos
    
    CALCULATE_PER_FRAME = 1
    CALCULATE_FINAL = False
    
    import json
    res_dict = {}
    res_dict['fvd']   = calculate_fvd  (videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, torch.device("cuda"))
    
    videos1 = origin_videos[:, cond_frames:cond_frames + total_pred_frames]
    videos2 = result_videos[:, cond_frames:cond_frames + total_pred_frames]
    
    res_dict['ssim']  = calculate_ssim (videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL)
    res_dict['psnr']  = calculate_psnr (videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL)
    res_dict['lpips'] = calculate_lpips(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, torch.device("cuda"))
    
    with open(json_path, "w") as f:
        json.dump(res_dict, f)

    end = timeit.default_timer()
    print(end - start, 'seconds')