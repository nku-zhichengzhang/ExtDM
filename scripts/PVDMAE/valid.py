# use LFAE to reconstruct validating videos and measure the loss in video domain
# using RegionMM

import argparse
from math import ceil

import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
import random
import json_tricks as json
from torch.utils.data import DataLoader
from data.video_dataset import VideoDataset, dataset2videos

import time

import einops

from torch.nn.functional import interpolate

from model.LFAE.flow_autoenc import FlowAE
from model.PVDM.autoencoder_vit import ViTAutoencoder as FlowAE

from utils.meter import AverageMeter
from utils.seed import setup_seed

if __name__ == "__main__":
    start = timeit.default_timer()
    cudnn.enabled = True
    cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Flow Autoencoder")
    parser.add_argument("--save-video", 
                        default=True)
    parser.add_argument("--cond_frames", 
                        type=int, 
                        default=10)
    parser.add_argument("--pred_frames", 
                        type=int, 
                        default=40)
    parser.add_argument("--num_videos", 
                        type=int, 
                        default=256)
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=64,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input_size", 
                        type=int, 
                        default=128,
                        help="height and width of videos")
    parser.add_argument("--random-seed", 
                        type=int, 
                        default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore_from", 
                        default="/mnt/sda/hjy/training_logs/cityscapes128/snapshots/RegionMM.pth")
    parser.add_argument("--config_path", 
                        default="./config/carla128.yaml")
    parser.add_argument("--log_dir", 
                        default="./logs_validation/flow/flowautoenc_video_carla")
    parser.add_argument("--data_dir", 
                        default="/mnt/sda/hjy/fdm/CARLA_Town_01_h5")
    parser.add_argument("--data_type", 
                        default="test")
    parser.add_argument("--num-workers", 
                        default=8)
    parser.add_argument("--gpu", 
                        default="0",
                        help="choose gpu device.")
    parser.add_argument("--fp16", 
                        default=False)
    args = parser.parse_args()

    setup_seed(args.random_seed)

    ckpt_dir = os.path.join(args.log_dir, "flowae_result")
    os.makedirs(ckpt_dir, exist_ok=True)

    json_path = os.path.join(ckpt_dir, "loss%d.json" % (args.num_videos))

    NUM_ITER = ceil (args.num_videos / args.batch_size)

    model = FlowAE(is_train=False, config=args.config_path)
    model.cuda()

    if os.path.isfile(args.restore_from):
        print("=> loading checkpoint '{}'".format(args.restore_from))
        checkpoint = torch.load(args.restore_from)
        model.load_state_dict(checkpoint)
        print("=> loaded checkpoint '{}'".format(args.restore_from))
    else:
        print("=> no checkpoint found at '{}'".format(args.restore_from))
        exit(-1)

    model.eval()

    valid_dataloader = data.DataLoader(VideoDataset(
                                    data_dir=args.data_dir,
                                    type=args.data_type, 
                                    image_size=args.input_size,
                                    num_frames=args.cond_frames+args.pred_frames,
                                    total_videos=args.num_videos,
                                    random_time=False
                                ),
                                batch_size=args.batch_size,
                                shuffle=False, 
                                num_workers=args.num_workers,
                                pin_memory=False
                            )

    batch_time = AverageMeter()
    data_time = AverageMeter()

    iter_end = timeit.default_timer()

    scale = 64/256

    out_loss = 0.0
    warp_loss = 0.0
    num_sample = 0.0
    l1_loss = torch.nn.L1Loss(reduction='sum')

    # b t c h w [0-1]
    origin_videos = []
    result_videos = []

    ii = iter(valid_dataloader)

    for iter in range(NUM_ITER):

        batch = next(ii)

        data_time.update(timeit.default_timer() - iter_end)

        total_vids, video_names = batch
        # (b t c h)/(b t h w c) -> (b t c h w)
        total_vids = dataset2videos(total_vids)
        origin_videos.append(total_vids)

        # real_vids 
        # torch.Size([b, 50, 3, 64, 64]) -> b c t h w
        # tensor(0.0431) tensor(0.9647)
        total_vids = total_vids.permute(0,2,1,3,4).contiguous()

        cond_vids = total_vids[:, :, :args.cond_frames, :, :]
        real_vids = total_vids[:, :, args.cond_frames:, :, :]

        # use first frame of each video as reference frame (vids: B C T H W)
        ref_imgs = cond_vids[:, :, -1, :, :].clone().detach()

        bs = real_vids.size(0)
        
        batch_time.update(timeit.default_timer() - iter_end)

        assert real_vids.size(2) == args.pred_frames

        # out_img_list = []
        # warped_img_list = []
        # warped_grid_list = []
        # conf_map_list = []

        # print("real_vids", real_vids.shape)
        real_vids_ = interpolate(real_vids, size=(16,256,256), mode='nearest', align_corners=None).cuda()
        # print("real_vids_", real_vids_.shape)

        with torch.no_grad():
            x_tilde = model(real_vids_*2-1)
        # print("x_tilde", x_tilde.shape)
        
        out_img_list_tensor = einops.rearrange(x_tilde.cpu(), '(b t) c h w -> b c t h w', b=bs)
        out_img_list_tensor = interpolate(out_img_list_tensor, size=(16,64,64), mode='nearest', align_corners=None)
        out_img_list_tensor = einops.rearrange(out_img_list_tensor, 'b c t h w -> t b c h w', b=bs)
        out_img_list_tensor = (out_img_list_tensor+1)/2
        print("out_img_list_tensor", out_img_list_tensor.shape)
        
        # for frame_idx in range(real_vids.size(2)):
        #     dri_imgs = real_vids[:, :, frame_idx, :, :]
        #     with torch.no_grad():
        #         model.set_train_input(ref_img=ref_imgs, dri_img=dri_imgs)
        #         model.forward()
        #     out_img_list.append(model.generated['prediction'].clone().detach())
        #     warped_img_list.append(model.generated['deformed'].clone().detach())
        #     warped_grid_list.append(model.generated['optical_flow'].clone().detach())
        #     conf_map_list.append(model.generated['occlusion_map'].clone().detach())

        # out_img_list_tensor = torch.stack(out_img_list, dim=0)
        # warped_img_list_tensor = torch.stack(warped_img_list, dim=0)
        # warped_grid_list_tensor = torch.stack(warped_grid_list, dim=0)
        # conf_map_list_tensor = torch.stack(conf_map_list, dim=0)

        from utils.visualize import LFAE_visualize
        LFAE_visualize(
            ground=real_vids,
            prediction=out_img_list_tensor,
            video_names=video_names,
            save_path=f"{args.log_dir}/flowae_result",
            save_num=8,
            image_size=ref_imgs.shape[-1]
        )
        

        import einops 
        tmp_result = torch.cat([
            einops.rearrange(cond_vids.cpu(),           'b c t h w -> b t c h w'), 
            einops.rearrange(out_img_list_tensor.cpu(), 't b c h w -> b t c h w')
            ], 
            dim=1
        )  
        result_videos.append(tmp_result)
        
        # out_loss += l1_loss(real_vids.permute(2, 0, 1, 3, 4).cpu(), out_img_list_tensor.cpu()).item()
        # warp_loss += l1_loss(real_vids.permute(2, 0, 1, 3, 4).cpu(), warped_img_list_tensor.cpu()).item()
        num_sample += bs

        iter_end = timeit.default_timer()

        print('Test:[{0}/{1}]\t''Time {batch_time.val:.3f}({batch_time.avg:.3f})'.format(iter, NUM_ITER, batch_time=batch_time))
            
    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)
    print(origin_videos.shape, result_videos.shape)

    print(torch.min(origin_videos), torch.max(origin_videos))
    print(torch.min(result_videos), torch.max(result_videos))

    torch.save(origin_videos, f'./{args.log_dir}/origin.pt')
    torch.save(result_videos, f'./{args.log_dir}/result.pt')
    
    
    # our gif        
    if args.save_video:      
        from utils.visualize import visualize
        visualize(
            save_path=f"{args.log_dir}/video_result",
            origin=origin_videos,
            result=result_videos,
            save_pic_num=8,
            grid_nrow=4,
            save_pic_row=True,
            save_gif=False,
            save_gif_grid=True,
            cond_frame_num=args.cond_frames,   
        )
        

    from metrics.calculate_fvd   import calculate_fvd1
    from metrics.calculate_psnr  import calculate_psnr1
    from metrics.calculate_ssim  import calculate_ssim1
    from metrics.calculate_lpips import calculate_lpips1

    device = torch.device("cuda")

    videos1 = origin_videos
    videos2 = result_videos

    # for i in range(2, 18):
    #     print(i, calculate_fvd1(videos1, videos2, device, mini_bs=i))
    
    print("[fvd    ]", calculate_fvd1(videos1, videos2, device, mini_bs=10))
    videos1 = videos1[:, args.cond_frames:]
    videos2 = videos2[:, args.cond_frames:]
    print("[ssim   ]", calculate_ssim1(videos1, videos2)[0])
    print("[psnr   ]", calculate_psnr1(videos1, videos2)[0])
    print("[lpips  ]", calculate_lpips1(videos1, videos2, device)[0])


    # print("loss for prediction: %.5f" % (out_loss/(num_sample*args.input_size*args.input_size*3)))
    # print("loss for warping: %.5f" % (warp_loss/(num_sample*args.input_size*args.input_size*3)))

    # res_dict = {}
    # res_dict["out_loss"] = out_loss/(num_sample*args.input_size*args.input_size*3)
    # res_dict["warp_loss"] = warp_loss/(num_sample*args.input_size*args.input_size*3)
    # with open(json_path, "w") as f:
    #     json.dump(res_dict, f)

    end = timeit.default_timer()
    print(end - start, 'seconds')



