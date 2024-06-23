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

start = timeit.default_timer()
BATCH_SIZE = 256 # 10
root_dir = './flowautoenc_video_kth/'
data_dir = "/mnt/sda/hjy/kth/processed"
GPU = "0"
postfix = ""
INPUT_SIZE = 64
COND_FRAMES = 10
PRED_FRAMES = 40
N_FRAMES = COND_FRAMES + PRED_FRAMES # 50 
NUM_VIDEOS = 256 # 16 #256
SAVE_VIDEO = True
NUM_ITER = NUM_VIDEOS // BATCH_SIZE
RANDOM_SEED = 1234 # 1234
MEAN = (0.0, 0.0, 0.0)
# the path to trained LFAE model
RESTORE_FROM = "/mnt/sda/hjy/flow_pretrained/kth64/snapshots/RegionMM.pth"
config_pth = "./config/kth64.yaml"

CKPT_DIR = os.path.join(root_dir, "flowae-res"+postfix)
# os.makedirs(CKPT_DIR, exist_ok=True)
json_path = os.path.join(CKPT_DIR, "loss%d%s.json" % (NUM_VIDEOS, postfix))
visualizer = Visualizer()
print(root_dir)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("config_path:", config_pth)
print(json_path)
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
    
    dataset_params = config['dataset_params']
    
    valid_dataset = VideoDataset(
        data_dir=dataset_params['root_dir'],
        type=dataset_params['valid_params']['type'], 
        image_size=dataset_params['frame_shape'],
        num_frames=dataset_params['valid_params']['cond_frames'] + dataset_params['valid_params']['pred_frames'], 
        total_videos=dataset_params['valid_params']['total_videos'],
        random_time=False,
    ) 


    batch_time = AverageMeter()
    data_time = AverageMeter()

    iter_end = timeit.default_timer()

    out_loss = 0.0
    warp_loss = 0.0
    num_sample = 0.0
    l1_loss = torch.nn.L1Loss(reduction='sum')

    # b t c h w [0-1]
    origin_videos = []
    result_videos = []
    
    # index = [66, 110, 127, 209, 226]
    index = [30,40,50,60,70,80,90,100]
    
    for idx in index:
        batch = valid_dataset[idx]
        
        os.makedirs(f'./flow_output/{idx}', exist_ok=True)

        data_time.update(timeit.default_timer() - iter_end)

        total_vids, _ = batch
        total_vids = total_vids.unsqueeze(0)
        total_vids = dataset2videos(total_vids)
        origin_videos.append(total_vids)

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
        warped_img_list = []
        warped_grid_list = []
        conf_map_list = []
        
        flow = []
        
        for frame_idx in range(COND_FRAMES+PRED_FRAMES):
            dri_imgs = total_vids[:, :, frame_idx, :, :]
            with torch.no_grad():
                model.set_train_input(ref_img=ref_imgs, dri_img=dri_imgs)
                model.forward()
            out_img_list.append(model.generated['prediction'].clone().detach())
            warped_img_list.append(model.generated['deformed'].clone().detach())
            warped_grid_list.append(model.generated['optical_flow'].clone().detach())
            conf_map_list.append(model.generated['occlusion_map'].clone().detach())
            
            driven = einops.rearrange(dri_imgs[0]*255, "c h w -> h w c").numpy()
            warped = model.generated['optical_flow'][0].clone().detach().cpu().numpy()
            # print('1', np.max(warped), np.min(warped))
            output = flow2fig(warped_grid=warped, grid_size=32, img_size=64)
            flow.append(output)
            # print('2', np.max(output), np.min(output))
            cv2.imwrite(f'./flow_output/{idx}/driven_{frame_idx}.png', driven)
            cv2.imwrite(f'./flow_output/{idx}/flow_{frame_idx}.png', output)

        out_img_list_tensor = torch.stack(out_img_list, dim=0)
        warped_img_list_tensor = torch.stack(warped_img_list, dim=0)
        warped_grid_list_tensor = torch.stack(warped_grid_list, dim=0)
        conf_map_list_tensor = torch.stack(conf_map_list, dim=0)
        
        origin = np.array(einops.rearrange(total_vids[0],'c t h w -> t h w c'))
        flow = np.array(flow)
        media.show_videos([origin, flow], fps=20)

        # out_img_list_tensor      [40, 8, 3, 64, 64] 
        # warped_img_list_tensor   [40, 8, 3, 64, 64]
        # warped_grid_list_tensor  [40, 8, 32, 32, 2]
        # conf_map_list_tensor     [40, 8, 1, 32, 32]

        
        tmp_result = torch.cat([
            einops.rearrange(cond_vids.cpu(),           'b c t h w -> b t c h w'), 
            einops.rearrange(out_img_list_tensor.cpu(), 't b c h w -> b t c h w')
            ], 
            dim=1
        )  
        result_videos.append(tmp_result)

        # out_loss += l1_loss(real_vids.permute(2, 0, 1, 3, 4).cpu(), out_img_list_tensor.cpu()).item()
        # warp_loss += l1_loss(real_vids.permute(2, 0, 1, 3, 4).cpu(), warped_img_list_tensor.cpu()).item()
        # num_sample += bs

        # if SAVE_VIDEO:
        #     for batch_idx in range(bs):
        #         #  LFAE 
        #         msk_size = ref_imgs.shape[-1] # h,w
        #         new_im_list = []
        #         for frame_idx in range(nf):
        #             # cond+real
        #             save_tar_img = sample_img(real_vids[:, :, frame_idx], batch_idx)
        #             # out_img_list_tensor
        #             save_out_img = sample_img(out_img_list_tensor[frame_idx], batch_idx)
        #             # warped_img_list_tensor
        #             save_warped_img = sample_img(warped_img_list_tensor[frame_idx], batch_idx)
        #             # warped_grid_list_tensor
        #             save_warped_grid = grid2fig(warped_grid_list_tensor[frame_idx, batch_idx].data.cpu().numpy(),grid_size=32, img_size=msk_size)
        #             # conf_map_list_tensor
        #             save_conf_map = conf_map_list_tensor[frame_idx, batch_idx].unsqueeze(dim=0)
        #             save_conf_map = save_conf_map.data.cpu()
        #             save_conf_map = F.interpolate(save_conf_map, size=real_vids.shape[3:5]).numpy()
        #             save_conf_map = np.transpose(save_conf_map, [0, 2, 3, 1])
        #             save_conf_map = np.array(save_conf_map[0, :, :, 0]*255, dtype=np.uint8)
        #             # save img_list
        #             new_im = Image.new('RGB', (msk_size * 5, msk_size))
        #             new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
        #             new_im.paste(Image.fromarray(save_out_img, 'RGB'), (msk_size, 0))
        #             new_im.paste(Image.fromarray(save_warped_img, 'RGB'), (msk_size * 2, 0))
        #             new_im.paste(Image.fromarray(save_warped_grid), (msk_size * 3, 0))
        #             new_im.paste(Image.fromarray(save_conf_map, "L"), (msk_size * 4, 0))
        #             new_im_list.append(new_im)
        #         video_name = "%s.gif" % (str(int(video_names[batch_idx])))
        #         imageio.mimsave(os.path.join(CKPT_DIR, video_name), new_im_list)
                
        #         # our gif   
        #         # pass   

        iter_end = timeit.default_timer()

    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)
    print(origin_videos.shape, origin_videos.shape)

    # # our gif        
    # if SAVE_VIDEO:      
    #     from v import visualize
    #     visualize(
    #         save_path=root_dir,
    #         origin=origin_videos,
    #         result=result_videos,
    #         save_pic_num=8,
    #         grid_nrow=4,
    #         save_pic_row=False,
    #         save_gif=False,
    #         cond_frame_num=10,   
    #     )

    # from fvd.calculate_fvd import calculate_fvd,calculate_fvd1
    # from fvd.calculate_psnr import calculate_psnr,calculate_psnr1
    # from fvd.calculate_ssim import calculate_ssim,calculate_ssim1
    # from fvd.calculate_lpips import calculate_lpips,calculate_lpips1

    # device = torch.device("cuda")

    # videos1 = origin_videos
    # videos2 = result_videos
    # print("[fvd    ]", calculate_fvd1(videos1, videos2, device))

    # for i in range(41):
    #     print(f"[fvd {COND_FRAMES+i}]", calculate_fvd1(videos1[:, :COND_FRAMES+i], videos2[:, :COND_FRAMES+i], device))


    # videos1 = videos1[:, COND_FRAMES:]
    # videos2 = videos2[:, COND_FRAMES:]
    # print("[ssim   ]", calculate_ssim1(videos1, videos2)[0])
    # print("[psnr   ]", calculate_psnr1(videos1, videos2)[0])
    # print("[lpips  ]", calculate_lpips1(videos1, videos2, device)[0])


    # print("loss for prediction: %.5f" % (out_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)))
    # print("loss for warping: %.5f" % (warp_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)))

    # res_dict = {}
    # res_dict["out_loss"] = out_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)
    # res_dict["warp_loss"] = warp_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)
    # with open(json_path, "w") as f:
    #     json.dump(res_dict, f)

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

