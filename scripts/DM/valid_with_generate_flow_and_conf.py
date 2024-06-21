import argparse

import torch
import torch.backends.cudnn as cudnn
import os
import timeit
import yaml
from tqdm import tqdm


from einops import rearrange, repeat

from torch.utils.data import DataLoader

from data.video_dataset import VideoDataset, dataset2videos
from utils.seed import setup_seed
from utils.misc import flow2fig, video_flow2fig, video_conf2fig

from metrics.calculate_fvd    import calculate_fvd,   calculate_fvd1, get_feats, calculate_fvd2
from metrics.calculate_psnr   import calculate_psnr,  calculate_psnr1,  calculate_psnr2
from metrics.calculate_ssim   import calculate_ssim,  calculate_ssim1,  calculate_ssim2
from metrics.calculate_lpips  import calculate_lpips, calculate_lpips1, calculate_lpips2

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
    parser.add_argument("--num_sample_video", 
                        type=int, 
                        default=10)
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
    parser.add_argument("--DM_arch", 
                        type=str)
    parser.add_argument("--Unet3D_arch", 
                        type=str)
    parser.add_argument("--dataset_path", 
                        type=str)
    parser.add_argument("--flowae_checkpoint",
                        default="/mnt/rhdd/zzc/data/video_prediction/flow_pretrained/better/smmnist64/snapshots/RegionMM.pth")
    parser.add_argument("--checkpoint", 
                        default="/home/ubuntu/zzc/code/videoprediction/EDM/logs_training/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5_STW_adaptor/snapshots/flowdiff.pth")
    parser.add_argument("--config", 
                        default="./config/smmnist64.yaml")
    parser.add_argument("--log_dir", 
                        default="./logs_validation/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5_STW_adaptor/")
    parser.add_argument("--random_time",
                        action="store_true",
                        help="set video of dataset starts from random_time.")
    parser.add_argument("--fp16", 
                        default=False)
    args = parser.parse_args()

    setup_seed(int(args.random_seed))

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.DM_arch == "VideoFlowDiffusion_multi":
        from model.BaseDM_adaptor.VideoFlowDiffusion_multi import FlowDiffusion
    elif args.DM_arch == "VideoFlowDiffusion_multi1248":
        from model.BaseDM_adaptor.VideoFlowDiffusion_multi1248 import FlowDiffusion
    elif args.DM_arch == "VideoFlowDiffusion_multi_w_ref":
        from model.BaseDM_adaptor.VideoFlowDiffusion_multi_w_ref import FlowDiffusion
    elif args.DM_arch == "VideoFlowDiffusion_multi_w_ref_u22":
        from model.BaseDM_adaptor.VideoFlowDiffusion_multi_w_ref_u22 import FlowDiffusion
    else:
        NotImplementedError()
        
    model = FlowDiffusion(
        config=config,
        pretrained_pth=args.flowae_checkpoint,
        is_train=False,
        Unet3D_architecture=args.Unet3D_arch
    )

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
    pred_frames = dataset_params['train_params']['pred_frames']

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
    origin_flows = []
    result_flows = []
    result_confs = []
    
    for i_iter, batch in enumerate(valid_dataloader):
        if i_iter >= NUM_ITER:
            break

        real_vids, _ = batch

        # (b t c h)/(b t h w c) -> (b t c h w)
        real_vids = dataset2videos(real_vids)
        real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')

        # generate n samples once a time
        real_vids = repeat(real_vids, 'b c t h w -> (b n) c t h w', n=args.num_sample_video)

        real_grids = get_flow(model, real_vids.cuda(), cond_frames)
        real_flows = video_flow2fig(video_warped_grid=real_grids, grid_size=real_grids.shape[-1], img_size=real_vids.shape[-1])

        origin_videos.append(real_vids)
        origin_flows.append(real_flows)

        pred_video = []
        pred_flow  = []
        pred_conf  = []

        i_real_vids = real_vids[:,:,:cond_frames]
        i_real_flow = real_flows[:,:,:cond_frames]
        
        for i_autoreg in range(NUM_AUTOREG):
            res = model.sample_one_video(cond_scale=1.0, real_vid=i_real_vids.cuda())
            i_pred_video = res['sample_out_vid'].clone().detach().cpu()
            i_pred_grid  = res['sample_vid_grid'].clone().detach().cpu()
            i_pred_flow  = video_flow2fig(video_warped_grid=i_pred_grid[:,:,-pred_frames:], grid_size=i_pred_grid.shape[-1], img_size=i_pred_video.shape[-1])
            i_pred_conf  = res['sample_vid_conf'].clone().detach().cpu()
            i_pred_conf  = video_conf2fig(video_warped=i_pred_conf[:,:,-pred_frames:], img_size=i_pred_video.shape[-1])
            # torch.Size([64, 2, 5, 32, 32])
            # torch.Size([64, 3, 5, 64, 64])
            print('i_pred_flow', i_pred_flow[:,:,-pred_frames:].shape)
            print('i_pred_conf', i_pred_conf[:,:,-pred_frames:].shape)
            print(f'[{i_autoreg}/{NUM_AUTOREG}] i_pred_video: {i_pred_video[:,:,-pred_frames:].shape}')
            pred_video.append(i_pred_video[:,:,-pred_frames:])
            pred_flow.append(i_pred_flow[:,:,-pred_frames:])
            pred_conf.append(i_pred_conf[:,:,-pred_frames:])
            i_real_vids = i_pred_video[:,:,-cond_frames:]
        pred_video = torch.cat(pred_video, dim=2)
        pred_flow  = torch.cat(pred_flow,  dim=2)
        pred_conf  = torch.cat(pred_conf,  dim=2)

        res_video = torch.cat([real_vids[:, :, :cond_frames], pred_video[:, :, :total_pred_frames]], dim=2)
        result_videos.append(res_video)  
        res_flow = torch.cat([real_flows[:, :, :cond_frames], pred_flow[:, :, :total_pred_frames]], dim=2)
        result_flows.append(res_flow)
        res_conf = torch.cat([torch.zeros_like(real_vids[:, :, :cond_frames]), pred_conf[:, :, :total_pred_frames]], dim=2)
        result_confs.append(res_conf)

    # (b n) c t h w
    origin_videos = torch.cat(origin_videos)
    # (b n) c t h w
    result_videos = torch.cat(result_videos)
    # (b n) c t h w
    origin_flows = torch.cat(origin_flows)/255.0
    # (b n) c t h w
    result_flows = torch.cat(result_flows)/255.0
    # (b n) c t h w
    result_confs = torch.cat(result_confs)

    # (b n) c t h w -> b n t c h w
    origin_videos = rearrange(origin_videos, '(b n) c t h w -> b n t c h w', n = args.num_sample_video)
    result_videos = rearrange(result_videos, '(b n) c t h w -> b n t c h w', n = args.num_sample_video)
    origin_flows  = rearrange(origin_flows,  '(b n) c t h w -> b n t c h w', n = args.num_sample_video)
    result_flows  = rearrange(result_flows,  '(b n) c t h w -> b n t c h w', n = args.num_sample_video)
    result_confs  = rearrange(result_confs,  '(b n) c t h w -> b n t c h w', n = args.num_sample_video)

    print(
        '#######',
        origin_videos.shape,
        result_videos.shape,
        origin_flows.shape,
        result_flows.shape,
        result_confs.shape,
        # abs_diff_videos.shape,
        # abs_diff_flows.shape
    )

    os.makedirs(f'./{args.log_dir}', exist_ok=True)
    torch.save(origin_videos[:,0], f'./{args.log_dir}/origin.pt')
    torch.save(origin_flows[:,0],  f'./{args.log_dir}/origin_flows.pt')
    for traj in tqdm(range(args.num_sample_video)):
        torch.save(result_videos[:, traj], f'./{args.log_dir}/result_{traj}.pt')
        torch.save(result_flows[:, traj],  f'./{args.log_dir}/result_flows_{traj}.pt')
        torch.save(result_confs[:, traj],  f'./{args.log_dir}/result_confs_{traj}.pt')

    from utils.visualize import visualize_ori_pre_flow_conf_save_pic

    # b n t c h w -> n=1 -> b t c h w
    
    for traj in tqdm(range(args.num_sample_video)):
        visualize_ori_pre_flow_conf_save_pic(
            save_path=f"{args.log_dir}/result_{traj}",
            origin=origin_videos[:,  0], 
            result=result_videos[:, traj], 
            flow=result_flows[:, traj], 
            conf=result_confs[:, traj], 
            cond_num=cond_frames,
            epoch_or_step_num=actual_step, 
        )