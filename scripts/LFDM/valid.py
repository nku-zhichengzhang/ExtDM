import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import yaml

from einops import rearrange

from torch.utils.data import DataLoader

from data.video_dataset import VideoDataset, dataset2videos
from utils.seed import setup_seed

from model.LFDM.video_flow_diffusion_model_pred import FlowDiffusion

if __name__ == "__main__":
    cudnn.enabled = True
    cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description="Diffusion")
    parser.add_argument("--total_pred_frames", 
                        type=int, 
                        default=10)
    parser.add_argument("--num_videos", 
                        type=int, 
                        default=256)
    parser.add_argument("--valid_batch_size", 
                        type=int, 
                        default=64,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ddim_sampling_steps", 
                        type=int, 
                        default=20)
    parser.add_argument("--random-seed", 
                        type=int, 
                        default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--dataset_path", 
                        type=str)
    parser.add_argument("--pred_frames", 
                        type=int)
    parser.add_argument("--flowae_checkpoint",
                        default="./logs_training/flow/kth64_test/snapshots/RegionMM.pth")
    parser.add_argument("--checkpoint", 
                        default="/mnt/sda/hjy/training_logs/cityscapes128/snapshots/RegionMM.pth")
    parser.add_argument("--config", 
                        default="./config/carla128.yaml")
    parser.add_argument("--log_dir", 
                        default="./logs_validation/flow/flowautoenc_video_carla")
    parser.add_argument("--random_time",
                        action="store_true",
                        help="set video of dataset starts from random_time.")
    parser.add_argument("--fp16", 
                        default=False)
    args = parser.parse_args()

    setup_seed(int(args.random_seed))

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config['diffusion_params']['model_params']['sampling_timesteps'] = int(args.ddim_sampling_steps)

    model = FlowDiffusion(
        config=config,
        pretrained_pth=args.flowae_checkpoint,
        is_train=False,
    )
    
    import torch.nn as nn

    def count_parameters(model):
        res = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"count_training_parameters: {res}")
        res = sum(p.numel() for p in model.parameters())
        print(f"count_all_parameters:      {res}")
    
    count_parameters(model)
    
    model.cuda()

    checkpoint = torch.load(args.checkpoint)
    model.diffusion.load_state_dict(checkpoint['diffusion'])

    model.eval()

    dataset_params = config['dataset_params']
    train_params = config['diffusion_params']['train_params']
    cond_frames = dataset_params['valid_params']['cond_frames']
    total_pred_frames = args.total_pred_frames
    # pred_frames = dataset_params['train_params']['pred_frames']
    # pred_frames = 5
    pred_frames=args.pred_frames

    valid_dataset = VideoDataset(
        data_dir=args.dataset_path,
        type=dataset_params['valid_params']['type'], 
        image_size=dataset_params['frame_shape'],
        num_frames=cond_frames + total_pred_frames, 
        total_videos=args.num_videos,
        random_time=args.random_time,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False, 
        num_workers=train_params['dataloader_workers'],
        pin_memory=True, 
        drop_last=False
    )
    
    from math import ceil
    NUM_ITER    = ceil(args.num_videos / args.valid_batch_size)
    NUM_AUTOREG = ceil(args.total_pred_frames / pred_frames)
    
    actual_step = int(ceil(checkpoint['example'] / train_params['batch_size']))
    
    # b t c h w [0-1]
    origin_videos = []
    result_videos = []
    fps = []
    
    for i_iter, batch in enumerate(valid_dataloader):
        if i_iter >= NUM_ITER:
            break
        
        real_vids, real_names = batch
        # (b t c h)/(b t h w c) -> (b t c h w)
        real_vids = dataset2videos(real_vids)
        real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')
        origin_videos.append(real_vids)
        pred_video = []

        i_real_vids = real_vids[:,:,:cond_frames]
        for i_autoreg in range(NUM_AUTOREG):
            start = time.time()
            i_pred_video = model.sample_one_video(cond_scale=1.0, real_vid=i_real_vids.cuda())['sample_out_vid'].clone().detach().cpu()
            end = time.time()
            tmp_fps = args.valid_batch_size*pred_frames/(end - start)
            fps.append(tmp_fps)
            print(f'FPS: {tmp_fps:.4f}')
            print(f'[{i_autoreg}/{NUM_AUTOREG}] i_pred_video: {i_pred_video[:,:,cond_frames:cond_frames+pred_frames].shape}')
            pred_video.append(i_pred_video[:,:,cond_frames:cond_frames+pred_frames])
            i_real_vids = i_pred_video[:,:,pred_frames:cond_frames+pred_frames]
        pred_video = torch.cat(pred_video, dim=2)

        res_video = torch.cat([real_vids[:, :, :cond_frames], pred_video[:, :, :total_pred_frames]], dim=2)
        result_videos.append(res_video)
    # print(fps)
    # print(f'avg FPS: {sum(fps[2:])/len(fps[2:]):.4f}')

    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)

    origin_videos = rearrange(origin_videos, 'b c t h w -> b t c h w')
    result_videos = rearrange(result_videos, 'b c t h w -> b t c h w')

    os.makedirs(f'./{args.log_dir}', exist_ok=True)
    torch.save(origin_videos, f'./{args.log_dir}/origin.pt')
    torch.save(result_videos, f'./{args.log_dir}/result.pt')
    
    from utils.visualize import visualize
    visualize(
        save_path=f"{args.log_dir}/result",
        origin=origin_videos,
        result=result_videos,
        save_pic_num=10,
        select_method='top',
        grid_nrow=5,
        save_gif_grid=True,
        save_pic_row=True,
        save_gif=True,
        epoch_or_step_num=actual_step, 
        cond_frame_num=cond_frames,
    )
    
    from metrics.calculate_fvd import calculate_fvd,calculate_fvd1
    from metrics.calculate_psnr import calculate_psnr,calculate_psnr1
    from metrics.calculate_ssim import calculate_ssim,calculate_ssim1
    from metrics.calculate_lpips import calculate_lpips,calculate_lpips1
    
    videos1 = origin_videos
    videos2 = result_videos
    
    fvd = calculate_fvd1(videos1, videos2, torch.device("cuda"), mini_bs=16)
    videos1 = videos1[:, cond_frames:]
    videos2 = videos2[:, cond_frames:]
    ssim = calculate_ssim1(videos1, videos2)[0]
    psnr = calculate_psnr1(videos1, videos2)[0]
    lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
    
    print("[fvd    ]", fvd)
    print("[ssim   ]", ssim)
    print("[psnr   ]", psnr)
    print("[lpips  ]", lpips)