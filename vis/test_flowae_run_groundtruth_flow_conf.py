# use LFAE to reconstruct testing videos and measure the loss in video domain
# using RegionMM

import argparse
import cv2
import einops 
import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
from PIL import Image
from data.video_dataset import VideoDataset, dataset2videos
import random
from model.LFAE.flow_autoenc import FlowAE
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm
import yaml

from utils.misc import flow2fig, conf2fig1
import mediapy as media 


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

def grid2fig(warped_grid, grid_size=32, img_size=256):
    dpi = 1000
    # plt.ioff()
    h_range = torch.linspace(-1, 1, grid_size)
    w_range = torch.linspace(-1, 1, grid_size)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).flip(2)
    flow_uv = grid.cpu().data.numpy()
    fig, ax = plt.subplots()
    grid_x, grid_y = warped_grid[..., 0], warped_grid[..., 1]
    plot_grid(flow_uv[..., 0], flow_uv[..., 1], ax=ax, color="lightgrey")
    plot_grid(grid_x, grid_y, ax=ax, color="C0")
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.set_size_inches(img_size/100, img_size/100)
    fig.set_dpi(100)
    out = fig2data(fig)[:, :, :3]
    plt.close()
    plt.cla()
    plt.clf()
    return out

start = timeit.default_timer()
COND_FRAMES = 10
PRED_FRAMES = 10
NUM_VIDEOS = 256 # 16 #256

# name = "bair_flow_conf"
# RESTORE_FROM = "/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained/BAIR/bair64_scale0.50/snapshots/RegionMM.pth"
# config_pth = "/home/u1120230288/zzc/code/video_prediction/EDM/config/bair64.yaml"

# name = "city_flow_conf"
# RESTORE_FROM = "/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained/Cityscapes/cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25/snapshots/RegionMM_0128_S150000.pth"
# config_pth = "/home/u1120230288/zzc/code/video_prediction/EDM/config/cityscapes128.yaml"

name = "kth_flow_conf"
RESTORE_FROM = "/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained/KTH/kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max40_2/snapshots/RegionMM_0256_S220000.pth"
config_pth = "/home/u1120230288/zzc/code/video_prediction/EDM/config/kth64.yaml"

# name = "smmnist_flow_conf"
# RESTORE_FROM = "/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained/SMMNIST/smmnist64_scale0.50/snapshots/RegionMM.pth"
# config_pth = "/home/u1120230288/zzc/code/video_prediction/EDM/config/smmnist64.yaml"

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
    
    dataset_params = config['dataset_params']
    
    valid_dataset = VideoDataset(
        data_dir=dataset_params['root_dir'],
        type=dataset_params['valid_params']['type'], 
        image_size=dataset_params['frame_shape'],
        num_frames=COND_FRAMES + PRED_FRAMES, 
        total_videos=NUM_VIDEOS,
        random_time=False,
    ) 

    batch_time = AverageMeter()
    data_time = AverageMeter()

    iter_end = timeit.default_timer()

    # b t c h w [0-1]
    origin_list=[]
    result_list=[]
    flow_list=[]
    conf_list=[]
    
    index = [i for i in range(len(valid_dataset))]
    
    for idx in tqdm(index):
        batch = valid_dataset[idx]
        
        os.makedirs(f'./flow_conf_output/{name}/', exist_ok=True)

        data_time.update(timeit.default_timer() - iter_end)

        total_vids, _ = batch
        total_vids = total_vids.unsqueeze(0)
        total_vids = dataset2videos(total_vids)

        # real_vids 
        # torch.Size([b, 50, 3, 64, 64]) -> b c t h w
        # tensor(0.0431) tensor(0.9647)
        total_vids = total_vids.permute(0,2,1,3,4).contiguous()

        cond_vids = total_vids[:, :, :COND_FRAMES, :, :]
        real_vids = total_vids[:, :, COND_FRAMES:, :, :]

        # use first frame of each video as reference frame (vids: B C T H W)
        ref_imgs = cond_vids[:, :, -1, :, :].clone().detach()

        bs = real_vids.size(0)

        batch_time.update(timeit.default_timer() - iter_end)

        nf = real_vids.size(2) # PRED_FRAMES
        assert nf == PRED_FRAMES

        out_img_list = []
        warped_grid_list = []
        conf_map_list = []
        
        for frame_idx in range(COND_FRAMES, COND_FRAMES+PRED_FRAMES):
            dri_imgs = total_vids[:, :, frame_idx, :, :]
            with torch.no_grad():
                model.set_train_input(ref_img=ref_imgs, dri_img=dri_imgs)
                model.forward()
            # 结果
            res = model.generated['prediction'][0].clone().detach().cpu()
            res = einops.rearrange(res, "c h w -> h w c")
            out_img_list.append(res)
            # 光流
            warped = model.generated['optical_flow'][0].clone().detach().cpu().numpy()
            warped = flow2fig(warped_grid=warped, grid_size=warped.shape[0], img_size=dri_imgs.shape[-1])
            warped_grid_list.append(torch.from_numpy(warped/255.0))
            # 置信度
            conf = model.generated['occlusion_map'][0].clone().detach().cpu().numpy()
            conf = einops.rearrange(conf, "c h w -> h w c")
            conf = conf2fig1(img=conf, img_size=dri_imgs.shape[-1])
            conf_map_list.append(torch.from_numpy(conf))

        out_img_list_tensor     = torch.stack(out_img_list, dim=0)
        warped_grid_list_tensor = torch.stack(warped_grid_list, dim=0)
        conf_map_list_tensor    = torch.stack(conf_map_list, dim=0)
        real_vids               = einops.rearrange(real_vids[0],'c t h w -> t h w c')

        # print(out_img_list_tensor.shape)
        # print(warped_grid_list_tensor.shape)
        # print(conf_map_list_tensor.shape)
        # print(real_vids.shape)
        # torch.Size([10, 64, 64, 3])
        # torch.Size([10, 64, 64, 3])
        # torch.Size([10, 64, 64, 3])
        # torch.Size([10, 64, 64, 3])

        origin_list.append(real_vids)
        result_list.append(out_img_list_tensor)
        flow_list.append(warped_grid_list_tensor)
        conf_list.append(conf_map_list_tensor)

        iter_end = timeit.default_timer()

    origin=torch.stack(origin_list)
    result=torch.stack(result_list)
    flow=torch.stack(flow_list)
    conf=torch.stack(conf_list)

    origin=einops.rearrange(origin, "b t h w c -> b t c h w")
    result=einops.rearrange(result, "b t h w c -> b t c h w")
    flow=einops.rearrange(flow, "b t h w c -> b t c h w")
    conf=einops.rearrange(conf, "b t h w c -> b t c h w")

    from utils.visualize import visualize_ori_pre_flow_conf

    # b t c h w
    visualize_ori_pre_flow_conf(
        save_path=f"./flow_conf_output/{name}",
        origin=origin, 
        result=result, 
        flow=flow, 
        conf=conf, 
        cond_num=0,
        epoch_or_step_num=0, 
    )

    # origin_videos = torch.cat(origin_videos)
    # result_videos = torch.cat(result_videos)
    # print(origin_videos.shape, origin_videos.shape)


    end = timeit.default_timer()
    print(end - start, 'seconds')




    


if __name__ == '__main__':
    main()

