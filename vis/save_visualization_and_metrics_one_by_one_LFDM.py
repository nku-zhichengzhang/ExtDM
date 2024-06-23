import os
import torch
from tqdm import tqdm
from utils.visualize import visualize
import scipy.stats as st
import numpy as np 
from einops import rearrange, repeat
from metrics.calculate_fvd    import calculate_fvd,   calculate_fvd1, get_feats, calculate_fvd2
from metrics.calculate_psnr   import calculate_psnr,  calculate_psnr1,  calculate_psnr2, calculate_psnr3
from metrics.calculate_ssim   import calculate_ssim,  calculate_ssim1,  calculate_ssim2
from metrics.calculate_lpips  import calculate_lpips, calculate_lpips1, calculate_lpips2, calculate_lpips3
import json

root = "/home/u1120230288/zzc/code/video_prediction/EDM/logs_validation/LFDM/"

# test_name = root + "BAIR/bair64_origin_LFDM_Batch64_lr2e-4_c2p5"
# test_name = root + "Cityscapes/cityscapes128_origin_LFDM_Batch64_lr2e-4_c2p5"
# test_name = root + "KTH/videoflowdiff_kth"
test_name = root + "SMMNIST/smmnist64_origin_LFDM_Batch32_lr2e-4_c10p5"
# test_name = root + "UCF101/ucf101_64_origin_LFDM_Batch16_lr2e-4_c4p8_scale0.50"

num_frames_cond = 10 # 2 10
num_frames_pred = 10 # 10 14 28 30 40
num_sample_video = 10
video_num = 256 # 256 100
mini_bs = 256
size = 64 # 64 128

def metrics_by_video(videos1, videos2, traj):

    with open(f'{test_name}/metrics_{traj}.csv', "w") as f:
            f.write('id,psnr\n')
            # f.write('id,psnr,ssim,lpips,fvd\n')
            
    for i in tqdm(range(video_num)):
        video1_ = videos1[i:i+1,num_frames_cond:]
        video2_ = videos2[i:i+1,num_frames_cond:]
        # metrics['fvd'] = calculate_fvd1(video1, video2, device)
        # video1 = video1[:,num_frames_cond:]
        # video2 = video2[:,num_frames_cond:]
        # metrics['ssim'] = calculate_ssim1(video1, video2)[0]
        metrics['psnr'] = calculate_psnr1(video1_, video2_)[0]
        # metrics['lpips'] = calculate_lpips1(video1, video2, device)[0]
        
        with open(f'{test_name}/metrics_{traj}.csv', "a") as f:
            f.write(f"{i},{metrics['psnr']:.6}\n")
            # f.write(f"{i},{metrics['psnr']:.6},{metrics['ssim']:.6},{metrics['lpips']:.6},{metrics['fvd']:.6}\n")

    # for traj in tqdm(range(num_sample_video)):
    #     with open(f'{test_name}/metrics_{traj}.csv', "w") as f:
    #             f.write('id,psnr,ssim,lpips\n')
    #             # f.write('id,psnr,ssim,lpips,fvd\n')
                
    #     for i in tqdm(range(video_num)):
    #         video1 = videos1[i:i+1,num_frames_cond:]
    #         video2 = videos2[traj, i:i+1,num_frames_cond:]
    #         # metrics['fvd'] = calculate_fvd1(video1, video2, device)
    #         # video1 = video1[:,num_frames_cond:]
    #         # video2 = video2[:,num_frames_cond:]
    #         metrics['ssim'] = calculate_ssim1(video1, video2)[0]
    #         metrics['psnr'] = calculate_psnr1(video1, video2)[0]
    #         metrics['lpips'] = calculate_lpips1(video1, video2, device)[0]
            
    #         with open(f'{test_name}/metrics_{traj}.csv', "a") as f:
    #             f.write(f"{i},{metrics['psnr']:.6},{metrics['ssim']:.6},{metrics['lpips']:.6}\n")
    #             # f.write(f"{i},{metrics['psnr']:.6},{metrics['ssim']:.6},{metrics['lpips']:.6},{metrics['fvd']:.6}\n")

# 按帧算
def metrics_by_frame(videos1, videos2, traj):

    # metrics['fvd'] = calculate_fvd(videos1, videos2, device)
    videos1_ = videos1[:,num_frames_cond:]
    videos2_ = videos2[:,num_frames_cond:]
    # metrics['ssim'] = calculate_ssim(videos1_, videos2_)
    metrics['psnr'] = calculate_psnr(videos1_, videos2_)
    # metrics['lpips'] = calculate_lpips(videos1_, videos2_, device)

    # for metrics_name in ['ssim', 'psnr', 'lpips']:
    for metrics_name in ['psnr']:
        with open(f'{test_name}/framewise-{metrics_name}-{traj}.csv', "w") as f:
            f.write(f'frame,value\n')
            for timestamp, value in metrics[metrics_name][metrics_name].items():
                t = int(timestamp.split('[')[-1].split(']')[0]) + (num_frames_cond if metrics_name != 'fvd' else 0)
                f.write(f'{t},{value}\n')

# # 按帧算
# def metrics_by_frame(videos1, videos2):

#     psnr_list  = []
#     ssim_list  = []
#     lpips_list = []

#     # for traj in tqdm(range(num_sample_video)):
#         # metrics['fvd'] = calculate_fvd(videos1, videos2, device)
#     videos1_ = videos1[:,num_frames_cond:].unsqueeze(0).repeat(len(videos2),1,1,1,1,1)
#     videos2_ = videos2[:,:,num_frames_cond:]
#     '''n b t c h w'''
#     videos1_ = rearrange(videos1_, 'n b t c h w-> b n t c h w')
#     videos2_ = rearrange(videos2_, 'n b t c h w-> b n t c h w')
#     for bid in range(len(videos1_)):
#         nt_psnr = calculate_psnr3(videos1_[bid], videos2_[bid])
#         n_psnr = np.mean(nt_psnr, axis=-1)
#         nid = np.argmax(n_psnr)
#         psnr_list.append(nt_psnr[nid])
#         # ssim_list.append (calculate_ssim (videos1_, videos2_))
#         # lpips_list.append(calculate_lpips(videos1_, videos2_, device))
#     t_psnr = np.mean(np.array(psnr_list), axis=0)
#     metrics['psnr']  = t_psnr
#     print(np.mean(t_psnr))
#     # metrics['ssim']  = np.max(np.mean(np.array(ssim_list) ,axis=1),axis=0)
#     # metrics['lpips'] = np.min(np.mean(np.array(lpips_list),axis=1),axis=0)

#     for metrics_name in ['psnr']:
#         with open(f'{test_name}/framewise-{metrics_name}.csv', "w") as f:
#             f.write(f'frame,value\n')
#             for timestamp, value in enumerate(metrics[metrics_name], start=num_frames_cond):
#                 f.write(f'{timestamp},{value}\n')

def show_videos(videos1, videos2, i):        
    
    visualize(
        save_path=f'{test_name}/result_{i}',
        origin=videos1,
        result=videos2,
        save_pic_num=256,
        select_method="top",
        grid_nrow=10,
        save_gif_grid=False,
        save_gif=False,
        save_pic_row=True,
        save_pic=True,
        skip_pic_num=2,
        cond_frame_num=num_frames_cond,   
    )

def metric_stuff(metric):
    avg_metric, std_metric = metric.mean().item(), metric.std().item()
    conf95_metric = avg_metric - float(st.norm.interval(confidence=0.95, loc=avg_metric, scale=st.sem(metric))[0])
    return avg_metric, std_metric, conf95_metric

def metrics_total(videos1, videos2):

    # b t c h w
    print(videos1.shape, torch.min(videos1), torch.max(videos1), torch.mean(videos1), torch.std(videos1))
    # n b t c h w
    print(videos2.shape, torch.min(videos2), torch.max(videos2), torch.mean(videos2), torch.std(videos2))

    ############################ fvd ################################
    # n b t c h w 
    fvd_list = []

    origin_feats = get_feats(videos1, torch.device("cuda"), mini_bs=mini_bs)
    result_feats = get_feats(rearrange(videos2, 'n b t c h w -> (b n) t c h w'), torch.device("cuda"), mini_bs=mini_bs)
    # avg_fvd = calculate_fvd2(origin_feats, result_feats)

    for traj in tqdm(range(num_sample_video)):
        result_feats_ = get_feats(videos2[traj], torch.device("cuda"), mini_bs=mini_bs)
        fvd_list.append(calculate_fvd2(origin_feats, result_feats_))
    
    print(fvd_list)
    fvd_traj_mean, fvd_traj_std, fvd_traj_conf95 = metric_stuff(np.array(fvd_list))
    
    ####################### psnr ssim lpips ###########################

    psnr_list  = []
    ssim_list  = []
    lpips_list = []
    select_scores = []

    for i_bs in tqdm(range(videos2.shape[1])):
        # [b t c h w] -> [n t c h w]
        a_origin_videos = videos1[   i_bs, num_frames_cond:].unsqueeze(0).repeat(len(videos2),1,1,1,1)
        # [n b t c h w] -> [n t c h w]
        a_result_videos = videos2[:, i_bs, num_frames_cond:]
        psnr_list.append (calculate_psnr2 (a_origin_videos, a_result_videos))
        ssim_list.append (calculate_ssim2 (a_origin_videos, a_result_videos))
        lpips_list.append(calculate_lpips2(a_origin_videos, a_result_videos, torch.device("cuda")))
        select_scores.append([np.abs(origin_feats[i_bs]-result_feats[i_bs*num_sample_video+i]).sum() for i in range(num_sample_video)])

    selected_index = np.argmin(np.array(select_scores), axis=-1)
    # best_videos = torch.from_numpy(np.array([videos2[selected_index[i], i] for i in range(selected_index.shape[0])]))
    best_feats = np.array([result_feats[i*num_sample_video + selected_index[i]] for i in range(selected_index.shape[0])])
    fvd = calculate_fvd2(origin_feats, best_feats)

    # print(psnr_list, ssim_list, lpips_list)

    avg_psnr, std_psnr, conf95_psnr    = metric_stuff(np.array(psnr_list) )
    avg_ssim, std_ssim, conf95_ssim    = metric_stuff(np.array(ssim_list) )
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

    print("[ fvd_best ]", fvd          )
    # print("[ fvd_all  ]", avg_fvd      )
    print("[ fvd_traj ]", fvd_traj_mean)
    print("[ ssim     ]", avg_ssim     )
    print("[ psnr     ]", avg_psnr     )
    print("[ lpips    ]", avg_lpips    )

    # visualize(
    #     save_path=f"{test_name}/result_best",
    #     origin=videos1,
    #     result=best_videos,
    #     save_pic_num=256,
    #     select_method='top',
    #     grid_nrow=10,
    #     save_gif_grid=True,
    #     save_pic_row=False,
    #     save_gif=False,
    #     save_pic=True, 
    #     skip_pic_num=1,
    #     epoch_or_step_num=0, 
    #     cond_frame_num=num_frames_cond,
    # )

def metrics_all(videos1, videos2):
    # metrics['fvd'] = calculate_fvd1(videos1, videos2, device)
    videos1 = videos1[:,num_frames_cond:]
    videos2 = videos2[:,num_frames_cond:]
    # metrics['ssim'] = calculate_ssim1(videos1, videos2)[0]
    metrics['psnr'] = calculate_psnr1(videos1, videos2)[0]
    # metrics['lpips'] = calculate_lpips1(videos1, videos2, device)[0]
    # print(metrics)
    print(metrics['psnr'])

# ps: pixel value for metrics should be in [0, 1]!

device = torch.device("cuda")

metrics = {}

# videos1 = torch.load(f'{test_name}_1000/origin.pt').cuda()[:video_num,:num_frames_cond+num_frames_pred]

# # 按每个轨迹单独计算
# # for i in tqdm(range(num_sample_video)):
# for i in range(num_sample_video):
#     # print("="*5, i, "="*5)
#     videos2 = torch.load(f'{test_name}_{(i+1)*1000}/result.pt').cuda()[:video_num,:num_frames_cond+num_frames_pred]
#     # metrics_by_video(videos1, videos2, i)
#     # metrics_by_frame(videos1, videos2, i)
#     # show_videos(videos1, videos2, i)
#     metrics_all(videos1, videos2)

# 只计算最好的
# videos2 = torch.load(f'{test_name}/result_best.pt').cuda()[:video_num,:num_frames_cond+num_frames_pred]
# metrics_by_video(videos1, videos2, 'best')
# # metrics_by_frame(videos1, videos2, 'best')
# show_videos(videos1, videos2, 'best')

# 每个轨迹合并后计算
# videos2 = torch.zeros((num_sample_video, *(videos1.shape)))
# for i in tqdm(range(num_sample_video)):
#     videos2[i] = torch.load(f'{test_name}/result_{i}.pt').cuda()[:video_num,:num_frames_cond+num_frames_pred]
# metrics_by_video(videos1, videos2, i)
# metrics_by_frame(videos1, videos2)
# show_videos(videos1, videos2, i)
# metrics_total(videos1, videos2)

def diversity(test_name, num_sample_video):
    videos1 = torch.load(f'{test_name}_1000/origin.pt')[:video_num,:num_frames_cond+num_frames_pred]
    if videos1.shape[2] == 3:
        videos1 = videos1[:,:,0]*0.299 + videos1[:,:,1]*0.587 + videos1[:,:,2]*0.114
    else:
        videos1 = videos1[:,:,0]

    all_processed_videos = torch.zeros(num_sample_video, video_num, num_frames_pred, size, size)
    
    for i in tqdm(range(10)):
        videos = torch.load(f'{test_name}_{(i+1)*1000}/result.pt')
        print(videos.shape)
        # b t c h w -> b t h w
        if videos.shape[2] == 3:
            videos = videos[:,:,0]*0.299 + videos[:,:,1]*0.587 + videos[:,:,2]*0.114
        else:
            videos = videos[:,:,0]
        videos = (videos - videos1)
        all_processed_videos[i] = videos[:,num_frames_cond:]
        del videos
    # n b t h w
    # all_processed_videos = all_processed_videos.std(dim=0)
    # # b t h w
    # all_processed_videos = all_processed_videos.mean(dim=(0,1,2,3))
    # # 1
    # print(all_processed_videos)
    # n b t h w
    all_processed_videos = all_processed_videos.std(dim=(0,1,2))
    # h w
    all_processed_videos = all_processed_videos.mean(dim=(0,1))
    # 1
    print(all_processed_videos)
    
diversity(test_name, num_sample_video)