from data.SMMNIST.stochastic_moving_mnist_edited import StochasticMovingMNIST
import mediapy as media

import cv2
import einops 
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import random
from model.LFAE.flow_autoenc import FlowAE
import yaml

from utils.misc import flow2fig

mnist_dir    = "/home/u1120230288/zzc/data/video_prediction/dataset/SMMNIST_h5"
name         = "smmnist_video2video"
INPUT_SIZE   = 64
COND_FRAMES  = 10
PRED_FRAMES  = 10
RESTORE_FROM = ""
config_pth   = ""

# cond_seq_len: 前10帧轨迹一致
# pred_seq_len: 后面10帧轨迹变化
# same_samples: 运动数字相同的视频数量
# diff_samples: 运动数字不同的视频数量
# 七个视频，最后一个为rdst视频，前六个为src视频
train_dataset = StochasticMovingMNIST(
    mnist_dir, train=True, num_digits=2,
    step_length=0.1, with_target=False,
    cond_seq_len=10, pred_seq_len=10, same_samples=6, diff_samples=1
)
# 取一组视频
a_video_samples = train_dataset[50]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(1234)

    model = FlowAE(is_train=False, config=config_pth)
    model.cuda()

    if os.path.isfile(RESTORE_FROM):
        print("=> loading checkpoint '{}'".format(RESTORE_FROM))
        checkpoint = torch.load(RESTORE_FROM)
        model.generator.load_state_dict(checkpoint['generator'])
        model.region_predictor.load_state_dict(checkpoint['region_predictor'])
        model.bg_predictor.load_state_dict(checkpoint['bg_predictor'])
        print("=> loaded checkpoint '{}'".format(RESTORE_FROM))
    else:
        print("=> no checkpoint found at '{}'".format(RESTORE_FROM))
        exit(-1)

    model.eval()

    setup_seed(1234)
    
    with open(config_pth) as f:
        config = yaml.safe_load(f)


    # 前面是不同的参考运动samples，最后一个是dist的sample
    ref_videos = a_video_samples
    dst_video = a_video_samples[-1]

    ref_videos = torch.stack(ref_videos).repeat(1,1,3,1,1)
    dst_videos = dst_video.unsqueeze(0).repeat(len(ref_videos),1,3,1,1)
    # print(ref_videos.shape, dst_videos.shape)
    # torch.Size([6, 20, 3, 64, 64])

    origin_batch = ref_videos.permute(0,2,1,3,4).contiguous()
    result_batch = dst_videos.permute(0,2,1,3,4).contiguous()

    cond_vids = origin_batch[:, :, :COND_FRAMES, :, :]
    pred_vids = result_batch[:, :, COND_FRAMES:, :, :]

    # use first frame of each video as reference frame (vids: B C T H W)
    ref_imgs = cond_vids[:, :, -1, :, :].cuda()

    real_grid_list = []
    real_conf_list = []
    
    for frame_idx in range(COND_FRAMES+PRED_FRAMES):
        if frame_idx < COND_FRAMES:
            dri_imgs = result_batch[:, :, frame_idx, :, :]
        else:
            dri_imgs = origin_batch[:, :, frame_idx, :, :]
        with torch.no_grad():
            model.set_train_input(ref_img=ref_imgs, dri_img=dri_imgs)
            model.forward()
        real_grid_list.append(model.generated["optical_flow"].permute(0, 3, 1, 2))
        real_conf_list.append(model.generated["occlusion_map"])

        # 输出生成的flow
        # driven = einops.rearrange(dri_imgs*255, "t c h w ->t h w c").numpy()
        # warped = model.generated['optical_flow'].clone().detach().cpu().numpy()
        # # print('1', np.max(warped), np.min(warped), warped.shape)
        # output = [flow2fig(warped_grid=warped[ii], grid_size=32, img_size=64) for ii in range(len(warped))]
        # output = np.stack(output)
        # # print('2', np.max(output), np.min(output), output.shape)
        # for ii in range(len(origin_batch)):
        #     os.makedirs(f'./video2video/{name}/motion_{ii}', exist_ok=True)
        #     print(driven[:,:,::-1].shape, output[:,:,::-1].shape)
        #     cv2.imwrite(f'./video2video/{name}/motion_{ii}/driven_{frame_idx}.png', driven[ii,:,:,::-1]) # RGB -> BGR
        #     cv2.imwrite(f'./video2video/{name}/motion_{ii}/flow_{frame_idx}.png',   output[ii,:,:,::-1]) # RGB -> BGR
        # exit()

    real_vid_grid = torch.stack(real_grid_list, dim=2)
    real_vid_conf = torch.stack(real_conf_list, dim=2)

    # print(real_vid_grid.shape, real_vid_grid.min(), real_vid_grid.max())
    # print(real_vid_conf.shape, real_vid_conf.min(), real_vid_conf.max())

    sample_vid_grid = real_vid_grid
    sample_vid_conf = real_vid_conf

    sample_out_img_list = []
    sample_warped_img_list = []
    
    for idx in range(sample_vid_grid.size(2)):
        sample_grid = sample_vid_grid[:, :, idx, :, :].permute(0, 2, 3, 1)
        sample_conf = sample_vid_conf[:, :, idx, :, :]
        # predict fake out image and fake warped image
        with torch.no_grad():
            generated_result = model.generator.forward_with_flow( 
                                            source_image=ref_imgs,
                                            optical_flow=sample_grid,
                                            occlusion_map=sample_conf)
        sample_out_img_list.append(generated_result["prediction"])
        
    sample_out_vid = torch.stack(sample_out_img_list, dim=2).cpu()
    # print(sample_out_vid.shape, sample_out_vid.min(), sample_out_vid.max())
    
    origin_videos = origin_batch
    result_videos = sample_out_vid

    origin_videos = einops.rearrange(origin_videos, 'b c t h w -> b t h w c').cpu()
    result_videos = einops.rearrange(result_videos, 'b c t h w -> b t h w c').cpu()

    print(origin_videos.shape, result_videos.shape)
    for i in range(origin_videos.shape[0]):
        print(origin_videos[i].numpy())
        media.write_video(f"./origin_{i}.gif", origin_videos[i].numpy(), fps=10)
    for i in range(result_videos.shape[0]):
        media.write_video(f"./result_{i}.gif", result_videos[i].numpy(), fps=10)

if __name__ == '__main__':
    main()
