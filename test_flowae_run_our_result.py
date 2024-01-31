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
from model.LFAE.util import Visualizer
import json_tricks as json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import yaml

from utils.misc import flow2fig
import mediapy as media 

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


name="bair64_not_onlyflow"

start = timeit.default_timer()
BATCH_SIZE = 256
data_dir = "/mnt/sda/hjy/bair/BAIR_h5"
GPU = "0"
postfix = ""
INPUT_SIZE = 64
COND_FRAMES = 2 # 10
PRED_FRAMES = 28 # 40
N_FRAMES = COND_FRAMES + PRED_FRAMES # 50 / 30
NUM_VIDEOS = 256 # 16 #256
SAVE_VIDEO = True
NUM_ITER = NUM_VIDEOS // BATCH_SIZE
RANDOM_SEED = 1234 # 1234
MEAN = (0.0, 0.0, 0.0)
# the path to trained LFAE model
# RESTORE_FROM = "/mnt/sda/hjy/flow_pretrained/kth64/snapshots/RegionMM.pth"
RESTORE_FROM = "/mnt/rhdd/zzc/data/video_prediction/flow_pretrained/better/bair64/snapshots/RegionMM.pth"
config_pth = f"./logs_training/diffusion/{name}/bair64.yaml"

visualizer = Visualizer()
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("config_path:", config_pth)
print("save video:", SAVE_VIDEO)


def sample_img(rec_img_batch, index):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array(MEAN)/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)


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
    
    valid_origin = torch.load(f'./logs_validation/diffusion/{name}/origin.pt')
    valid_result = torch.load(f'./logs_validation/diffusion/{name}/result.pt')

    batch_time = AverageMeter()
    data_time = AverageMeter()

    iter_end = timeit.default_timer()

    
    # index = [1,17,24,26,112,176,203,204,223,232]
    index = [2,5,27,30,52,61,71,84,134,222]
    
    for idx in index:
        origin_batch = valid_origin[idx]
        result_batch = valid_result[idx]
        
        os.makedirs(f'./flow_output/{name}/{idx}', exist_ok=True)

        data_time.update(timeit.default_timer() - iter_end)

        origin_batch = origin_batch.unsqueeze(0)
        result_batch = result_batch.unsqueeze(0)

        # real_vids b t c h w -> b c t h w
        # tensor(0.0431) tensor(0.9647)
        origin_batch = origin_batch.permute(0,2,1,3,4).contiguous()
        result_batch = result_batch.permute(0,2,1,3,4).contiguous()

        cond_vids = origin_batch[:, :, :COND_FRAMES, :, :]
        pred_vids = result_batch[:, :, COND_FRAMES:, :, :]

        # use first frame of each video as reference frame (vids: B C T H W)
        ref_imgs = cond_vids[:, :, -1, :, :].clone().detach()

        batch_time.update(timeit.default_timer() - iter_end)

        flow = []
        
        for frame_idx in range(COND_FRAMES+PRED_FRAMES):
            dri_imgs = result_batch[:, :, frame_idx, :, :]
            with torch.no_grad():
                model.set_train_input(ref_img=ref_imgs, dri_img=dri_imgs)
                model.forward()
            
            driven = einops.rearrange(dri_imgs[0]*255, "c h w -> h w c").numpy()
            warped = model.generated['optical_flow'][0].clone().detach().cpu().numpy()
            print('1', np.max(warped), np.min(warped), warped.shape)
            output = flow2fig(warped_grid=warped, grid_size=32, img_size=64)
            flow.append(output)
            print('2', np.max(output), np.min(output), output.shape)
            cv2.imwrite(f'./flow_output/{name}/{idx}/driven_{frame_idx}.png', driven[:,:,::-1]) # RGB -> BGR
            cv2.imwrite(f'./flow_output/{name}/{idx}/flow_{frame_idx}.png', output[:,:,::-1]) # RGB -> BGR
        
        video = np.array(einops.rearrange(result_batch[0],'c t h w -> t h w c'))
        flow = np.array(flow)
        media.show_videos([, flow], fps=20)

        iter_end = timeit.default_timer()

    end = timeit.default_timer()
    print(end - start, 'seconds')


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
    
if __name__ == '__main__':
    main()

