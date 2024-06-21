import argparse

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
import yaml
from tqdm import tqdm
import scipy.stats as st

from einops import rearrange, repeat

from torch.utils.data import DataLoader

from data.video_dataset import VideoDataset, dataset2videos
from utils.seed import setup_seed

from metrics.calculate_fvd    import calculate_fvd,   calculate_fvd1, get_feats, calculate_fvd2
from metrics.calculate_psnr   import calculate_psnr,  calculate_psnr1,  calculate_psnr2
from metrics.calculate_ssim   import calculate_ssim,  calculate_ssim1,  calculate_ssim2
from metrics.calculate_lpips  import calculate_lpips, calculate_lpips1, calculate_lpips2

def metric_stuff(metric):
    avg_metric, std_metric = metric.mean().item(), metric.std().item()
    conf95_metric = avg_metric - float(st.norm.interval(confidence=0.95, loc=avg_metric, scale=st.sem(metric))[0])
    return avg_metric, std_metric, conf95_metric


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
    parser.add_argument("--estimate_occlusion_map",
                        action="store_true")
    parser.add_argument("--random_time",
                        action="store_true",
                        help="set video of dataset starts from random_time.")
    parser.add_argument("--fp16", 
                        default=False)
    args = parser.parse_args()

    setup_seed(int(args.random_seed))

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config["flow_params"]["model_params"]["generator_params"]["pixelwise_flow_predictor_params"]["estimate_occlusion_map"] = args.estimate_occlusion_map

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
    
    for i_iter, batch in enumerate(valid_dataloader):
        if i_iter >= NUM_ITER:
            break

        real_vids, real_names = batch
        # (b t c h)/(b t h w c) -> (b t c h w)
        real_vids = dataset2videos(real_vids)
        real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')

        # generate n samples once a time
        real_vids = repeat(real_vids, 'b c t h w -> (b n) c t h w', n=args.num_sample_video)
        
        pred_video = []

        i_real_vids = real_vids[:,:,:cond_frames]
        # print(111, i_real_vids.shape, real_vids.shape)
        
        for i_autoreg in range(NUM_AUTOREG):
            i_pred_video = model.sample_one_video(cond_scale=1.0, real_vid=i_real_vids.cuda())['sample_out_vid'].clone().detach().cpu()
            print(f'[{i_autoreg+1}/{NUM_AUTOREG}] i_pred_video: {i_pred_video[:,:,-pred_frames:].shape}')
            pred_video.append(i_pred_video[:,:,-pred_frames:])
            i_real_vids = i_pred_video[:,:,-cond_frames:]
            torch.cuda.empty_cache()


        pred_video = torch.cat(pred_video, dim=2)
        res_video = torch.cat([real_vids[:, :, :cond_frames], pred_video[:, :, :total_pred_frames]], dim=2)
        
        # torch.Size([32, 3, 12, 64, 64]) torch.Size([32, 3, 12, 64, 64])
        # print(real_vids.shape, res_video.shape)
        # (b n) c t h w
        origin_videos.append(real_vids)
        # # (b n) c t h w
        result_videos.append(res_video)

        print(f'[{i_iter+1}/{NUM_ITER}] test videos generated.')
        torch.cuda.empty_cache()

    # (b n) c t h w
    origin_videos = torch.cat(origin_videos)
    # (b n) c t h w
    result_videos = torch.cat(result_videos)
    
    # (b n) c t h w -> b n t c h w
    origin_videos = rearrange(origin_videos, '(b n) c t h w -> b n t c h w', n = args.num_sample_video)
    result_videos = rearrange(result_videos, '(b n) c t h w -> b n t c h w', n = args.num_sample_video)

    print(origin_videos.shape, result_videos.shape)

    ############################ fvd ################################
    # b n t c h w 
    fvd_list = []

    origin_feats = get_feats(          origin_videos[:, 0]                           , torch.device("cuda"), mini_bs=16)
    result_feats = get_feats(rearrange(result_videos, 'b n t c h w -> (b n) t c h w'), torch.device("cuda"), mini_bs=16)
    # avg_fvd = calculate_fvd2(origin_feats, result_feats)

    for traj in tqdm(range(args.num_sample_video), desc='fvd_feature'):
        result_feats_ = get_feats(result_videos[:, traj], torch.device("cuda"), mini_bs=16)
        fvd_list.append(calculate_fvd2(origin_feats, result_feats_))
    
    # print(avg_fvd, fvd_list)
    print(fvd_list)
    fvd_traj_mean, fvd_traj_std, fvd_traj_conf95 = metric_stuff(np.array(fvd_list))
    
    ####################### psnr ssim lpips ###########################

    psnr_list  = []
    ssim_list  = []
    lpips_list = []
    select_scores = []

    for i_bs in tqdm(range(len(result_videos))):
        # get [n t c h w]
        a_origin_videos = origin_videos[i_bs, :, cond_frames:]
        a_result_videos = result_videos[i_bs, :, cond_frames:]
        psnr_list.append (calculate_psnr2 (a_origin_videos, a_result_videos))
        ssim_list.append (calculate_ssim2 (a_origin_videos, a_result_videos))
        lpips_list.append(calculate_lpips2(a_origin_videos, a_result_videos, torch.device("cuda")))
        select_scores.append([ np.abs(origin_feats[i_bs]-result_feats[i_bs*args.num_sample_video+i]).sum() for i in range(args.num_sample_video) ])

    fvd = []

    selected_index = np.argmin(np.array(select_scores), axis=-1)
    best_videos = torch.from_numpy(np.array([result_videos[i, selected_index[i]] for i in range(selected_index.shape[0])]))
    best_feats = get_feats(best_videos, torch.device("cuda"), mini_bs=16)
    fvd = calculate_fvd2(origin_feats, best_feats)

    # print(psnr_list, ssim_list, lpips_list)
    # print(selected_index)

    avg_psnr, std_psnr, conf95_psnr    = metric_stuff(np.array(psnr_list))
    avg_ssim, std_ssim, conf95_ssim    = metric_stuff(np.array(ssim_list))
    avg_lpips, std_lpips, conf95_lpips = metric_stuff(np.array(lpips_list))

    vid_metrics = {
        'psnr':  avg_psnr,  'psnr_std':  std_psnr,  'psnr_conf95':  conf95_psnr,
        'ssim':  avg_ssim,  'ssim_std':  std_ssim,  'ssim_conf95':  conf95_ssim,
        'lpips': avg_lpips, 'lpips_std': std_lpips, 'lpips_conf95': conf95_lpips,
        # 'fvd_all':   avg_fvd,
        'fvd_best':  fvd,
        'fvd_traj_mean': fvd_traj_mean, 'fvd_traj_std': fvd_traj_std, 'fvd_traj_conf95': fvd_traj_conf95
    }

    print(vid_metrics)

    end = timeit.default_timer()
    delta = end - start

    print("[ fvd_best ]", fvd          )
    # print("[ fvd_all  ]", avg_fvd      )
    print("[ fvd_traj ]", fvd_traj_mean)
    print("[ ssim     ]", avg_ssim     )
    print("[ psnr     ]", avg_psnr     )
    print("[ lpips    ]", avg_lpips    )
    print("[ time     ]", delta , 'seconds.')

    os.makedirs(f'./{args.log_dir}', exist_ok=True)

    with open(f'{args.log_dir}/metrics.txt', 'w') as f:
        f.write(f"[ fvd_best ] {fvd          }\n")
        # f.write(f"[ fvd_all  ] {avg_fvd      }\n")
        f.write(f"[ fvd_traj ] {fvd_traj_mean}\n")
        f.write(f"[ ssim     ] {avg_ssim     }\n")
        f.write(f"[ psnr     ] {avg_psnr     }\n")
        f.write(f"[ lpips    ] {avg_lpips    }\n")
        f.write(f"[ time     ] {delta        } seconds.\n")
    
    ################################# save tensor ##############################
    # b n t c h w

    torch.save(origin_videos[:,0].clone(), f'./{args.log_dir}/origin.pt')

    torch.save(best_videos.clone(), f'./{args.log_dir}/result_best.pt')

    for traj in tqdm(range(args.num_sample_video)):
        torch.save(result_videos[:, traj].clone(), f'./{args.log_dir}/result_{traj}.pt')

    ################################# save visualize ###########################
    # b n t c h w

    from utils.visualize import visualize

    print(origin_videos[:,0].shape, best_videos.shape)

    visualize(
        save_path=f"{args.log_dir}/result_best",
        origin=origin_videos[:,0],
        result=best_videos,
        save_pic_num=10,
        select_method='linspace',
        grid_nrow=4,
        save_gif_grid=True,
        save_pic_row=False,
        save_gif=False,
        save_pic=True, 
        skip_pic_num=1,
        epoch_or_step_num=actual_step, 
        cond_frame_num=cond_frames,
    )

    # for traj in random.sample(range(args.num_sample_video), min(args.num_sample_video, 3)):
    #     visualize(
    #         save_path=f"{args.log_dir}/result_{traj}",
    #         origin=origin_videos[:,0],
    #         result=result_videos[:, traj],
    #         save_pic_num=256,
    #         select_method='top',
    #         grid_nrow=10,
    #         save_gif_grid=True,
    #         save_pic_row=False,
    #         save_gif=False,
    #         save_pic=True, 
    #         skip_pic_num=1,
    #         epoch_or_step_num=actual_step, 
    #         cond_frame_num=cond_frames,
    #     )