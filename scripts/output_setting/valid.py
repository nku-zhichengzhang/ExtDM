import argparse

import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
import random
import yaml

from einops import rearrange

from torch.utils.data import DataLoader

from data.video_dataset import VideoDataset, dataset2videos
from model.LFAE.flow_autoenc import FlowAE
from utils.meter import AverageMeter
from utils.seed import setup_seed

from utils.misc import flow2fig, video_flow2fig

# from model.DM.video_flow_diffusion_model_pred_condframe_temp import FlowDiffusion
# from model.BaseDM_adaptor.VideoFlowDiffusion1 import FlowDiffusion
from model.BaseDM_adaptor.VideoFlowDiffusion_multi1248 import FlowDiffusion
from model.BaseDM_adaptor.VideoFlowDiffusion_multi_w_ref_u22 import FlowDiffusion
# from model.BaseDM_adaptor.VideoFlowDiffusion_multi_w_ref import FlowDiffusion
# from model.BaseDM_adaptor.VideoFlowDiffusion_multi import FlowDiffusion

def get_flow(model, real_vid, cond_frame_num):
    # b c t h w 
    # -> 64, 2, 5, 32, 32
    real_grid_list = []
    with torch.no_grad():
        ref_img = real_vid[:,:,cond_frame_num-1,:,:]
        source_region_params = model.region_predictor(ref_img)
        for idx in range(real_vid.shape[2]):
            driving_region_params = model.region_predictor(real_vid[:, :, idx, :, :])
            bg_params = model.bg_predictor(ref_img, real_vid[:, :, idx, :, :])
            generated = model.generator(ref_img,
                                        driving_region_params,
                                        source_region_params,
                                        bg_params)
            generated.update({'source_region_params': source_region_params,
                                'driving_region_params': driving_region_params})
            real_grid_list.append(generated["optical_flow"].permute(0, 3, 1, 2))
    real_vid_grid = torch.stack(real_grid_list, dim=2)
    ret = real_vid_grid.cpu()
    return ret

if __name__ == "__main__":
    start = timeit.default_timer()
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
    parser.add_argument("--random-seed", 
                        type=int, 
                        default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--flowae_checkpoint",
                        default="/mnt/rhdd/zzc/data/video_prediction/flow_pretrained/better/smmnist64/snapshots/RegionMM.pth")
    parser.add_argument("--checkpoint", 
                        default="/home/ubuntu/zzc/code/videoprediction/EDM/logs_training/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5_STW_adaptor/snapshots/flowdiff.pth")
    parser.add_argument("--config", 
                        default="./config/smmnist64.yaml")
    parser.add_argument("--log_dir", 
                        default="./logs_validation/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5_STW_adaptor/")
    parser.add_argument("--random_time", 
                        type=bool,
                        default=False,
                        help="set video of dataset starts from random_time.")
    parser.add_argument("--device_ids", 
                        default="0",
                        help="choose gpu device.")
    parser.add_argument("--fp16", 
                        default=False)
    args = parser.parse_args()

    setup_seed(int(args.random_seed))

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
    total_pred_frames = args.total_pred_frames
    pred_frames = dataset_params['train_params']['pred_frames']

    valid_dataset = VideoDataset(
        data_dir=dataset_params['root_dir'],
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
    origin_flows = []
    result_flows = []
    
    for i_iter, batch in enumerate(valid_dataloader):
        if i_iter >= NUM_ITER:
            break

        real_vids, real_names = batch
        # (b t c h)/(b t h w c) -> (b t c h w)
        real_vids = dataset2videos(real_vids)
        real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')
        
        real_grids = get_flow(model, real_vids.cuda(), cond_frames)
        real_flows = video_flow2fig(video_warped_grid=real_grids, grid_size=real_grids.shape[-1], img_size=real_vids.shape[-1])

        origin_videos.append(real_vids)
        origin_flows.append(real_flows)
        pred_video = []
        pred_flow  = []

        i_real_vids = real_vids[:,:,:cond_frames]
        i_real_flow = real_flows[:,:,:cond_frames]
        
        for i_autoreg in range(NUM_AUTOREG):
            res = model.sample_one_video(cond_scale=1.0, real_vid=i_real_vids.cuda())
            i_pred_video = res['sample_out_vid'].clone().detach().cpu()
            i_pred_grid  = res['sample_vid_grid'].clone().detach().cpu()
            i_pred_flow  = video_flow2fig(video_warped_grid=i_pred_grid[:,:,-pred_frames:], grid_size=i_pred_grid.shape[-1], img_size=real_vids.shape[-1])
            # torch.Size([64, 2, 5, 32, 32])
            # torch.Size([64, 3, 5, 64, 64])
            print('i_pred_flow', i_pred_flow[:,:,-pred_frames:].shape)
            print(f'[{i_autoreg}/{NUM_AUTOREG}] i_pred_video: {i_pred_video[:,:,-pred_frames:].shape}')
            pred_video.append(i_pred_video[:,:,-pred_frames:])
            pred_flow.append(i_pred_flow[:,:,-pred_frames:])
            i_real_vids = i_pred_video[:,:,-cond_frames:]

        pred_video = torch.cat(pred_video, dim=2)
        pred_flow  = torch.cat(pred_flow,  dim=2)

        res_video = torch.cat([real_vids[:, :, :cond_frames], pred_video[:, :, :total_pred_frames]], dim=2)
        result_videos.append(res_video)
        
        res_flow = torch.cat([real_flows[:, :, :cond_frames], pred_flow[:, :, :total_pred_frames]], dim=2)
        result_flows.append(res_flow)

    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)
    origin_flows = torch.cat(origin_flows)/255.0
    result_flows = torch.cat(result_flows)/255.0
    
    # print('origin_videos', origin_videos.shape)
    # print('result_videos', result_videos.shape)
    # print('result_flows', result_flows.shape)

    origin_videos = rearrange(origin_videos, 'b c t h w -> b t c h w')
    result_videos = rearrange(result_videos, 'b c t h w -> b t c h w')
    origin_flows  = rearrange(origin_flows,  'b c t h w -> b t c h w')
    result_flows  = rearrange(result_flows,  'b c t h w -> b t c h w')
    abs_diff_videos = torch.sqrt(torch.sum((result_videos - origin_videos)**2, dim=2)/3).unsqueeze(2).repeat(1,1,3,1,1)
    abs_diff_flows  = torch.sqrt(torch.sum((result_flows - origin_flows)**2, dim=2)/3).unsqueeze(2).repeat(1,1,3,1,1)
    
    print(
        '#######',
        origin_videos.shape,
        result_videos.shape,
        origin_flows.shape,
        result_flows.shape,
        abs_diff_videos.shape,
        abs_diff_flows.shape
    )

    os.makedirs(f'./{args.log_dir}', exist_ok=True)
    torch.save(origin_videos, f'./{args.log_dir}/origin.pt')
    torch.save(result_videos, f'./{args.log_dir}/result.pt')
    torch.save(origin_flows,  f'./{args.log_dir}/origin_flows.pt')
    torch.save(result_flows,  f'./{args.log_dir}/result_flows.pt')
    
    # from utils.visualize import visualize_ori_pre_flow_diff
    # visualize_ori_pre_flow_diff(
    #     save_path=f"{args.log_dir}/result",
    #     origin=origin_videos, 
    #     result=result_videos, 
    #     origin_flow=origin_flows, 
    #     result_flow=result_flows, 
    #     video_diff=abs_diff_videos, 
    #     flow_diff=abs_diff_flows, 
    #     epoch_or_step_num=actual_step, 
    #     cond_frame_num=cond_frames, 
    #     skip_pic_num=1
    # )

    from utils.visualize import visualize
    visualize(
        save_path=f"{args.log_dir}/result",
        origin=origin_videos,
        result=result_videos,
        save_pic_num=256,
        select_method='top',
        grid_nrow=4,
        save_gif_grid=False,
        save_pic_row=True,
        save_gif=False,
        save_pic=True,
        epoch_or_step_num=actual_step, 
        cond_frame_num=cond_frames,
        skip_pic_num=1,
    )

    visualize(
        save_path=f"{args.log_dir}/flow_result",
        origin=origin_flows,
        result=result_flows,
        save_pic_num=256,
        select_method='top',
        grid_nrow=4,
        save_gif_grid=False,
        save_pic_row=True,
        save_gif=False,
        save_pic=True,
        epoch_or_step_num=actual_step, 
        cond_frame_num=cond_frames,
        skip_pic_num=1,
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

    end = timeit.default_timer()
    print(end - start, 'seconds')