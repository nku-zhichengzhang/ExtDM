# train a LFAE
# this code is based on RegionMM (MRAA): https://github.com/snap-research/articulated-animation
import os.path
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from model.LFAE.model import ReconstructionModel
from model.LFAE.util import Visualizer
from model.LFAE.sync_batchnorm import DataParallelWithCallback
from data.two_frames_dataset import DatasetRepeater
import timeit
import imageio
import math
import random

import wandb
import datetime

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from utils.meter import AverageMeter
from model.LFAE.flow_autoenc import FlowAE

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

def sample_img(rec_img_batch, index):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array((0.0, 0.0, 0.0))/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)

def train(config, generator, region_predictor, bg_predictor, checkpoint, log_dir, train_dataset, valid_dataset, device_ids):
    print(config)

    train_params = config['train_params']

    optimizer = torch.optim.Adam(list(generator.parameters()) +
                                 list(region_predictor.parameters()) +
                                 list(bg_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.login()
    wandb.init(
        project="EDM",
        config={
            "learning_rate": train_params['lr'],
            "epochs": train_params['max_epochs'],
        },
        name=f"{config['experiment_name']}_{now}",
        dir=f"./wandb/{config['experiment_name']}_{now}"
    )

    start_epoch = 0
    start_step = 0
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        if config["set_start"]:
            start_step = int(math.ceil(ckpt['example'] / config['train_params']['batch_size']))
            start_epoch = ckpt['epoch']

            print("ckpt['example']", ckpt['example'])
            print("start_step", start_step)

            if config['dataset_params']['frame_shape'] == 64:
                ckpt['generator']['pixelwise_flow_predictor.down.weight'] = generator.pixelwise_flow_predictor.down.weight
                ckpt['region_predictor']['pixelwise_flow_predictor.down.weight'] = region_predictor.pixelwise_flow_predictor.down.weight
        
        if config['dataset_params']['frame_shape'] == 64:
            ckpt['generator']['pixelwise_flow_predictor.down.weight'] = generator.pixelwise_flow_predictor.down.weight
            ckpt['region_predictor']['down.weight'] = region_predictor.down.weight
        
        generator.load_state_dict(ckpt['generator'])
        region_predictor.load_state_dict(ckpt['region_predictor'])
        bg_predictor.load_state_dict(ckpt['bg_predictor'])
        
        if 'optimizer' in list(ckpt.keys()):
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except:
                optimizer.load_state_dict(ckpt['optimizer'].state_dict())

    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        train_dataset = DatasetRepeater(train_dataset, train_params['num_repeats'])

    train_dataloader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=False)

    model = ReconstructionModel(region_predictor, bg_predictor, generator, train_params)

    visualizer = Visualizer(**config['visualizer_params'])

    if torch.cuda.is_available():
        if ('use_sync_bn' in train_params) and train_params['use_sync_bn']:
            model = DataParallelWithCallback(model, device_ids=device_ids)
        else:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

    # rewritten by nhm
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_losses = AverageMeter()
    losses_perc = AverageMeter()
    losses_equiv_shift = AverageMeter()
    losses_equiv_affine = AverageMeter()

    cnt = 0
    epoch_cnt = start_epoch
    actual_step = start_step
    final_step = config["num_step_per_epoch"] * train_params["max_epochs"]

    while actual_step < final_step:
        iter_end = timeit.default_timer()

        for i_iter, x in enumerate(train_dataloader):
            actual_step = int(start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)
            optimizer.zero_grad()
            losses, generated = model(x)
            loss_values = [val.mean() for val in losses.values()]
            loss = sum(loss_values)
            loss.backward()
            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            bs = x['source'].size(0)
            total_losses.update(loss, bs)
            losses_perc.update(loss_values[0], bs)
            losses_equiv_shift.update(loss_values[1], bs)
            losses_equiv_affine.update(loss_values[2], bs)

            if actual_step % train_params["print_freq"] == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'loss_perc {loss_perc.val:.4f} ({loss_perc.avg:.4f})\n'
                      'loss_shift {loss_shift.val:.4f} ({loss_shift.avg:.4f})\t'
                      'loss_affine {loss_affine.val:.4f} ({loss_affine.avg:.4f})\t'
                      'time {batch_time.val:.4f} ({batch_time.avg:.4f})'
                    .format(
                    cnt, actual_step, final_step,
                    loss=total_losses,
                    loss_perc=losses_perc,
                    loss_shift=losses_equiv_shift,
                    loss_affine=losses_equiv_affine,
                    batch_time=batch_time
                ))

                wandb.log({
                    "actual_step": actual_step,
                    "loss": total_losses.val, 
                    "loss_perc": losses_perc.val,
                    "loss_shift": losses_equiv_shift.val,
                    "loss_affine": losses_equiv_affine.val,
                    "batch_time": batch_time.val
                })

            if actual_step % train_params['save_img_freq'] == 0:
                save_image = visualizer.visualize(x['driving'], x['source'], generated, index=0)
                save_name = 'B' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") \
                            + '_' + str(x["frame"][0][0].item()) + '_to_' + str(x["frame"][0][1].item()) +'.png'
                save_file = os.path.join(config["imgshots"], save_name)
                imageio.imsave(save_file, save_image)

                wandb.log({"save_img": wandb.Image(save_image)})

            if actual_step % config["save_ckpt_freq"] == 0 and cnt != 0:
                print('taking snapshot...')
                checkpoint_save_path = os.path.join(
                        config["snapshots"], 
                        'RegionMM_' + format(train_params["batch_size"], "04d") +'_S' + format(actual_step, "06d") + '.pth'
                    )
                torch.save({
                    'example': actual_step * train_params["batch_size"],
                    'epoch': epoch_cnt,
                    'generator': generator.state_dict(),
                    'bg_predictor': bg_predictor.state_dict(),
                    'region_predictor': region_predictor.state_dict(),
                    'optimizer': optimizer.state_dict()
                    },
                    checkpoint_save_path
                )
                
            if actual_step % train_params["update_ckpt_freq"] == 0 and cnt != 0:
                print('updating snapshot...')
                checkpoint_save_path=os.path.join(config["snapshots"], 'RegionMM.pth')
                torch.save({'example': actual_step * train_params["batch_size"],
                            'epoch': epoch_cnt,
                            'generator': generator.state_dict(),
                            'bg_predictor': bg_predictor.state_dict(),
                            'region_predictor': region_predictor.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           checkpoint_save_path)
                metrics = valid(config, valid_dataloader, checkpoint_save_path, log_dir, actual_step)
                wandb.log(metrics)

            if actual_step >= final_step:
                break

            cnt += 1

        scheduler.step()
        epoch_cnt += 1
        # print lr
        print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer.param_groups[0]["lr"]))

    print('save the final model...')
    torch.save({'example': actual_step * train_params["batch_size"],
                'epoch': epoch_cnt,
                'generator': generator.state_dict(),
                'bg_predictor': bg_predictor.state_dict(),
                'region_predictor': region_predictor.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(config["snapshots"],
                            'RegionMM_' + format(train_params["batch_size"], "04d") +
                            '_S' + format(actual_step, "06d") + '.pth'))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def valid(config, valid_dataloader, checkpoint_save_path, log_dir, epoch_num):

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(1234)

    model = FlowAE(is_train=False, config=config)
    model.cuda()

    checkpoint = torch.load(checkpoint_save_path)
    model.generator.load_state_dict(checkpoint['generator'])
    model.region_predictor.load_state_dict(checkpoint['region_predictor'])
    model.bg_predictor.load_state_dict(checkpoint['bg_predictor'])

    model.eval()

    from math import ceil

    NUM_ITER = ceil(config['valid_dataset_params']['total_videos'] / config['train_params']['batch_size'])

    cond_frames = config['valid_dataset_params']['cond_frames']
    pred_frames = config['valid_dataset_params']['pred_frames']

    origin_videos = []
    result_videos = []

    for i_iter, batch in enumerate(valid_dataloader):
        if i_iter >= NUM_ITER:
            break

        total_vids, video_names = batch

        origin_videos.append(total_vids)

        # real_vids 
        # torch.Size([b, 50, 3, 64, 64]) -> b c t h w
        # tensor(0.0431) tensor(0.9647)
        total_vids = total_vids.permute(0,2,1,3,4).contiguous()

        cond_vids = total_vids[:, :, :cond_frames, :, :]
        real_vids = total_vids[:, :, cond_frames:, :, :]

        # use first frame of each video as reference frame (vids: B C T H W)
        ref_imgs = cond_vids[:, :, -1, :, :].clone().detach()

        bs = real_vids.size(0)

        nf = real_vids.size(2) 
        assert nf == pred_frames

        out_img_list = []
        warped_img_list = []
        warped_grid_list = []
        conf_map_list = []

        for frame_idx in range(nf):
            dri_imgs = real_vids[:, :, frame_idx, :, :]
            with torch.no_grad():
                model.set_train_input(ref_img=ref_imgs, dri_img=dri_imgs)
                model.forward()
            out_img_list.append(model.generated['prediction'].clone().detach())
            warped_img_list.append(model.generated['deformed'].clone().detach())
            warped_grid_list.append(model.generated['optical_flow'].clone().detach())
            conf_map_list.append(model.generated['occlusion_map'].clone().detach())

        out_img_list_tensor = torch.stack(out_img_list, dim=0)
        warped_img_list_tensor = torch.stack(warped_img_list, dim=0)
        warped_grid_list_tensor = torch.stack(warped_grid_list, dim=0)
        conf_map_list_tensor = torch.stack(conf_map_list, dim=0)

        for batch_idx in range(bs):
            #  LFAE 
            msk_size = ref_imgs.shape[-1] # h,w
            new_im_list = []
            for frame_idx in range(nf):
                # cond+real
                save_tar_img = sample_img(real_vids[:, :, frame_idx], batch_idx)
                # out_img_list_tensor
                save_out_img = sample_img(out_img_list_tensor[frame_idx], batch_idx)
                # warped_img_list_tensor
                save_warped_img = sample_img(warped_img_list_tensor[frame_idx], batch_idx)
                # warped_grid_list_tensor
                save_warped_grid = grid2fig(warped_grid_list_tensor[frame_idx, batch_idx].data.cpu().numpy(),grid_size=32, img_size=msk_size)
                # conf_map_list_tensor
                save_conf_map = conf_map_list_tensor[frame_idx, batch_idx].unsqueeze(dim=0)
                save_conf_map = save_conf_map.data.cpu()
                save_conf_map = F.interpolate(save_conf_map, size=real_vids.shape[3:5]).numpy()
                save_conf_map = np.transpose(save_conf_map, [0, 2, 3, 1])
                save_conf_map = np.array(save_conf_map[0, :, :, 0]*255, dtype=np.uint8)
                # save img_list
                new_im = Image.new('RGB', (msk_size * 5, msk_size))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(save_out_img, 'RGB'), (msk_size, 0))
                new_im.paste(Image.fromarray(save_warped_img, 'RGB'), (msk_size * 2, 0))
                new_im.paste(Image.fromarray(save_warped_grid), (msk_size * 3, 0))
                new_im.paste(Image.fromarray(save_conf_map, "L"), (msk_size * 4, 0))
                new_im_list.append(new_im)
            video_name = "%s.gif" % (str(int(video_names[batch_idx])))
            os.makedirs(os.path.join(log_dir, "flowae-res"),exist_ok=True)
            imageio.mimsave(os.path.join(log_dir, "flowae-res", video_name), new_im_list)

            break

        # out_img_list_tensor      [40, 8, 3, 64, 64] 
        # warped_img_list_tensor   [40, 8, 3, 64, 64]
        # warped_grid_list_tensor  [40, 8, 32, 32, 2]
        # conf_map_list_tensor     [40, 8, 1, 32, 32]

        import einops 
        tmp_result = torch.cat([
            einops.rearrange(cond_vids.cpu(),           'b c t h w -> b t c h w'), 
            einops.rearrange(out_img_list_tensor.cpu(), 't b c h w -> b t c h w')
            ], 
            dim=1
        )  
        result_videos.append(tmp_result)

        print('Test:[{0}/{1}]\t'.format(i_iter, NUM_ITER))
            

    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)
    print(origin_videos.shape, origin_videos.shape)

    from utils.visualize import visualize
    visualize(
        save_path=f"{log_dir}/video",
        origin=origin_videos,
        result=result_videos,
        save_pic_num=8,
        grid_nrow=4,
        save_pic_row=False,
        save_gif=False,
        epoch_num=epoch_num, 
        cond_frame_num=cond_frames,   
    )

    from metrics.calculate_fvd   import calculate_fvd1
    from metrics.calculate_psnr  import calculate_psnr1
    from metrics.calculate_ssim  import calculate_ssim1
    from metrics.calculate_lpips import calculate_lpips1
    device = torch.device("cuda")
    videos1 = origin_videos
    videos2 = result_videos

    fvd   = calculate_fvd1(videos1, videos2, device, mini_bs=4)
    videos1 = videos1[:, cond_frames:]
    videos2 = videos2[:, cond_frames:]
    ssim  = calculate_ssim1(videos1, videos2)[0]
    psnr  = calculate_psnr1(videos1, videos2)[0]
    lpips = calculate_lpips1(videos1, videos2, device)[0]

    # print("[fvd    ]", fvd)
    # print("[ssim   ]", ssim)
    # print("[psnr   ]", psnr)
    # print("[lpips  ]", lpips)

    return {
        'metrics/fvd': fvd,
        'metrics/ssim': ssim,
        'metrics/psnr': psnr,
        'metrics/lpips': lpips
    }
