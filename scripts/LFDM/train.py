import torch
import os.path
import numpy as np
import math
import sys

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from torch.optim.lr_scheduler import LambdaLR
from utils.lr_scheduler import LambdaLinearScheduler

from data.two_frames_dataset import DatasetRepeater

from utils.misc import grid2fig, conf2fig
from utils.meter import AverageMeter
from utils.visualize import sample_img
from utils.seed import setup_seed

from PIL import Image
import timeit
import wandb
from einops import rearrange
import imageio

import torch.backends.cudnn as cudnn
from data.video_dataset import VideoDataset, dataset2videos

from model.LFDM.video_flow_diffusion_model_pred import FlowDiffusion

def train(
        config, 
        dataset_params,
        train_params,
        log_dir, 
        checkpoint,
        device_ids
    ):

    print(config)
    print(device_ids)
    
    model = FlowDiffusion(
        config=config,
        pretrained_pth=config['flowae_checkpoint'],
        is_train=True,
    )

    model.cuda()

    train_dataset = VideoDataset(
        data_dir=dataset_params['root_dir'],
        type=dataset_params['train_params']['type'], 
        image_size=dataset_params['frame_shape'],
        num_frames=dataset_params['train_params']['cond_frames'] + dataset_params['train_params']['pred_frames'],
    )
    
    valid_dataset = VideoDataset(
        data_dir=dataset_params['root_dir'],
        type=dataset_params['valid_params']['type'], 
        image_size=dataset_params['frame_shape'],
        num_frames=dataset_params['valid_params']['cond_frames'] + dataset_params['valid_params']['pred_frames'], 
        total_videos=dataset_params['valid_params']['total_videos'],
    )

    # 计算一个 epoch 有多少 step 
    steps_per_epoch = math.ceil(train_params['num_repeats'] * len(train_dataset) / float(train_params['batch_size']))
    # 多少 step 保存一次模型
    save_ckpt_freq = train_params['save_ckpt_freq']
    print("save ckpt freq:", save_ckpt_freq)
    
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        train_dataset = DatasetRepeater(train_dataset, train_params['num_repeats'])

    optimizer = torch.optim.AdamW(
        model.diffusion.parameters(), 
        lr=train_params['lr'],
        betas=(0.9, 0.999),
        eps=1.0e-08,
        weight_decay=0.0,
        amsgrad=False
    )

    start_epoch = 0
    start_step = 0

    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            print("=> loading checkpoint '{}'".format(checkpoint))
            ckpt = torch.load(checkpoint)
            if config["set_start"]:
                start_step = int(math.ceil(ckpt['example'] / train_params['batch_size']))
                start_epoch = ckpt['epoch']

                print("start_step", start_step)
                print("start_epoch", start_epoch)

            model_ckpt = model.diffusion.state_dict()
            for name, _ in model_ckpt.items():
                model_ckpt[name].copy_(ckpt['diffusion'][name])
            model.diffusion.load_state_dict(model_ckpt)
            print("=> loaded checkpoint '{}'".format(checkpoint))
            if "optimizer" in list(ckpt.keys()):
                optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint))
    else:
        print("NO checkpoint found!")

    scheduler = MultiStepLR(optimizer, last_epoch=start_step - 1, **train_params['scheduler_param'])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_params['batch_size'],
        shuffle=True, 
        num_workers=train_params['dataloader_workers'],
        pin_memory=True,
        drop_last=False
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=train_params['valid_batch_size'],
        shuffle=False, 
        num_workers=train_params['dataloader_workers'],
        pin_memory=True, 
        drop_last=False
    )

    # if torch.cuda.is_available():
    #     model.to(args.device_ids[0])
    # Not set model to be train mode! Because pretrained flow autoenc need to be eval

    if torch.cuda.is_available():
        if ('use_sync_bn' in train_params) and train_params['use_sync_bn']:
            model = DataParallelWithCallback(model, device_ids=device_ids)
        else:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_rec = AverageMeter()
    losses_warp = AverageMeter()

    cnt = 0
    epoch_cnt = start_epoch
    actual_step = start_step
    final_step = steps_per_epoch * train_params["max_epochs"]

    print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer.param_groups[0]["lr"]))
    
    while actual_step < final_step:
        iter_end = timeit.default_timer()

        for i_iter, batch in enumerate(train_dataloader):
            actual_step = int(start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)

            real_vids, real_names = batch
            # (b t c h)/(b t h w c) -> (b t c h w)
            real_vids = dataset2videos(real_vids)
            real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')

            # print(real_vids.shape)
            # torch.Size([8, 20, 3, 64, 64])
            # torch.Size([bs, c, length, h, w])
            ref_imgs = real_vids[:, :, dataset_params['train_params']['cond_frames']-1, :, :].clone().detach()
            real_vid = real_vids[:, :, :dataset_params['train_params']['cond_frames'] + dataset_params['train_params']['pred_frames']].cuda()
            
            optimizer.zero_grad()
            ret = model(real_vid)

            # print(ret['loss'].mean().item())
            # print(ret['rec_loss'].mean().item())
            # print(ret['rec_warp_loss'].mean().item())

            loss_ = ret['loss'].mean()
            loss_rec = ret['rec_loss'].mean()
            loss_rec_warp = ret['rec_warp_loss'].mean()

            if model.module.only_use_flow:
                loss_.backward()
            else:
                (loss_ + loss_rec + loss_rec_warp).backward()
                
            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            bs = real_vids.size(0)
            losses.update(loss_.item(), bs)
            losses_rec.update(loss_rec.item(), bs)
            losses_warp.update(loss_rec_warp.item(), bs)

            if actual_step % train_params["print_freq"] == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.7f} ({loss.avg:.7f})\t'
                      'loss_rec {loss_rec.val:.4f} ({loss_rec.avg:.4f})\t'
                      'loss_warp {loss_warp.val:.4f} ({loss_warp.avg:.4f})'
                    .format(
                    cnt, actual_step, final_step,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_rec=losses_rec,
                    loss_warp=losses_warp,
                ))

                wandb.log({
                    "actual_step": actual_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": losses.val, 
                    "loss_rec": losses_rec.val,
                    "loss_warp": losses_warp.val,
                    "batch_time": batch_time.avg
                })

            if actual_step % train_params['save_img_freq'] == 0:
                msk_size = ref_imgs.shape[-1]
                save_src_img = sample_img(ref_imgs)
                save_tar_img = sample_img(real_vids[:, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2, :, :])
                save_real_out_img = sample_img(ret['real_out_vid'][:, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2, :, :])
                save_real_warp_img = sample_img(ret['real_warped_vid'][:, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2, :, :])
                save_fake_out_img = sample_img(ret['fake_out_vid'][:, :, dataset_params['train_params']['pred_frames']//2, :, :])
                save_fake_warp_img = sample_img(ret['fake_warped_vid'][:, :, dataset_params['train_params']['pred_frames']//2, :, :])
                save_real_grid = grid2fig(ret['real_vid_grid'][0, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=12, img_size=msk_size)
                save_fake_grid = grid2fig(ret['fake_vid_grid'][0, :, dataset_params['train_params']['pred_frames']//2].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=12, img_size=msk_size)
                save_real_conf = conf2fig(ret['real_vid_conf'][0, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2], img_size=dataset_params['frame_shape'])
                save_fake_conf = conf2fig(ret['fake_vid_conf'][0, :, dataset_params['train_params']['pred_frames']//2], img_size=dataset_params['frame_shape'])
                new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                
                # imgshot
                # -------------------------------------------------------
                # | src | real_out  | real_warp | real_grid | real_conf |
                # -------------------------------------------------------
                # | tar | fake_out  | fake_warp | fake_grid | fake_conf |
                # -------------------------------------------------------
                
                new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                
                new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                new_im.paste(Image.fromarray(save_fake_out_img, 'RGB'), (msk_size, msk_size))
                
                new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size * 2, 0))
                new_im.paste(Image.fromarray(save_fake_warp_img, 'RGB'), (msk_size * 2, msk_size))
                
                new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                
                new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                
                new_im_name = 'B' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") \
                              + '_' + format(real_names[0], "06d") + ".png"
                new_im_file = os.path.join(config["imgshots"], new_im_name)
                new_im.save(new_im_file)
                wandb.log({
                    "save_img": wandb.Image(new_im)
                })

            if actual_step % train_params['save_vid_freq'] == 0:
                print("saving video...")
                # num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                save_src_img = sample_img(ref_imgs)
                for nf in range(dataset_params['train_params']['cond_frames'], dataset_params['train_params']['cond_frames'] + dataset_params['train_params']['pred_frames']):
                    save_tar_img = sample_img(real_vids[:, :, nf, :, :])
                    save_real_out_img = sample_img(ret['real_out_vid'][:, :, nf, :, :])
                    save_real_warp_img = sample_img(ret['real_warped_vid'][:, :, nf, :, :])
                    save_fake_out_img = sample_img(ret['fake_out_vid'][:, :, nf - dataset_params['train_params']['cond_frames'], :, :])
                    save_fake_warp_img = sample_img(ret['fake_warped_vid'][:, :, nf - dataset_params['train_params']['cond_frames'], :, :])
                    save_real_grid = grid2fig(
                        ret['real_vid_grid'][0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=12, img_size=msk_size)
                    save_fake_grid = grid2fig(
                        ret['fake_vid_grid'][0, :, nf - dataset_params['train_params']['cond_frames']].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=12, img_size=msk_size)
                    save_real_conf = conf2fig(ret['real_vid_conf'][0, :, nf], img_size=dataset_params['frame_shape'])
                    save_fake_conf = conf2fig(ret['fake_vid_conf'][0, :, nf - dataset_params['train_params']['cond_frames']], img_size=dataset_params['frame_shape'])
                    new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                    
                    # videoshot
                    # -------------------------------------------------------
                    # | src | real_out | real_warp | real_grid | real_conf |
                    # -------------------------------------------------------
                    # | tar | fake_out | fake_warp | fake_grid | fake_conf |
                    # -------------------------------------------------------
                    
                    new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                    
                    new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_fake_out_img, 'RGB'), (msk_size, msk_size))
                    
                    new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size * 2, 0))
                    new_im.paste(Image.fromarray(save_fake_warp_img, 'RGB'), (msk_size * 2, msk_size))
                    
                    new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                    new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                    
                    new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                    new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                new_vid_name = 'B' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") \
                                + '_' + format(real_names[0], "06d") + ".gif"
                new_vid_file = os.path.join(config["vidshots"], new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)


            # save model
            if actual_step % train_params['save_ckpt_freq'] == 0 and cnt != 0:
                print('taking snapshot ...')
                torch.save({
                        'example': actual_step * train_params["batch_size"],
                        'epoch': epoch_cnt,
                        'diffusion': model.module.diffusion.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(config["snapshots"],'flowdiff_' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") + '.pth'))

            # update saved model
            if actual_step % train_params['update_ckpt_freq'] == 0 and cnt != 0:
                print('updating saved snapshot ...')
                checkpoint_save_path=os.path.join(config["snapshots"], 'flowdiff.pth')
                torch.save({
                        'example': actual_step * train_params["batch_size"],
                        'epoch': epoch_cnt,
                        'diffusion': model.module.diffusion.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    checkpoint_save_path)
                metrics = valid(config, valid_dataloader, checkpoint_save_path, log_dir, actual_step)
                wandb.log(metrics)

            if actual_step >= final_step:
                break

            cnt += 1
            # 按 step 进行 warmup 策略
            scheduler.step()
            
        epoch_cnt += 1
        print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer.param_groups[0]["lr"]))

    print('save the final model ...')
    torch.save({
        'example': actual_step * train_params["batch_size"],
        'diffusion': model.module.diffusion.state_dict(),
        'optimizer': optimizer.state_dict()
        },
        os.path.join(config["snapshots"], 'flowdiff_' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") + '.pth'))


def valid(config, valid_dataloader, checkpoint_save_path, log_dir, actual_step):
    
    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(1234)

    model = FlowDiffusion(
        config=config,
        pretrained_pth=config['flowae_checkpoint'],
        is_train=False,
    )
    model.cuda()

    checkpoint = torch.load(checkpoint_save_path)
    model.diffusion.load_state_dict(checkpoint['diffusion'])

    model.eval()

    dataset_params = config['dataset_params']
    train_params = config['diffusion_params']['train_params']
    cond_frames = dataset_params['valid_params']['cond_frames']
    total_pred_frames = dataset_params['valid_params']['pred_frames']
    pred_frames = dataset_params['train_params']['pred_frames']

    from math import ceil
    NUM_ITER    = ceil(dataset_params['valid_params']['total_videos'] / train_params['valid_batch_size'])
    NUM_AUTOREG = ceil(total_pred_frames / pred_frames)
    
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

        origin_videos.append(real_vids)
        pred_video = []

        i_real_vids = real_vids[:,:,:cond_frames]
        
        for i_autoreg in range(NUM_AUTOREG):
            i_pred_video = model.sample_one_video(cond_scale=1.0, real_vid=i_real_vids.cuda())['sample_out_vid'].clone().detach().cpu()
            print(f'[{i_autoreg}/{NUM_AUTOREG}] i_pred_video: {i_pred_video[:,:,-pred_frames:].shape}')
            pred_video.append(i_pred_video[:,:,-pred_frames:])
            i_real_vids = i_pred_video[:,:,-cond_frames:]

        pred_video = torch.cat(pred_video, dim=2)

        res_video = torch.cat([real_vids[:, :, :cond_frames], pred_video[:, :, :total_pred_frames]], dim=2)
        result_videos.append(res_video)

    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)

    origin_videos = rearrange(origin_videos, 'b c t h w -> b t c h w')
    result_videos = rearrange(result_videos, 'b c t h w -> b t c h w')
    
    from utils.visualize import visualize
    visualize(
        save_path=f"{log_dir}/video_result",
        origin=origin_videos,
        result=result_videos,
        save_pic_num=8,
        select_method='linspace',
        grid_nrow=4,
        save_gif_grid=False,
        save_gif=True,
        save_pic_row=False,
        save_pic=False,
        epoch_or_step_num=actual_step, 
        cond_frame_num=cond_frames,
    )

    from metrics.calculate_fvd import calculate_fvd,calculate_fvd1
    from metrics.calculate_psnr import calculate_psnr,calculate_psnr1
    from metrics.calculate_ssim import calculate_ssim,calculate_ssim1
    from metrics.calculate_lpips import calculate_lpips,calculate_lpips1
    
    fvd = calculate_fvd1(origin_videos, result_videos, torch.device("cuda"), mini_bs=16)
    videos1 = origin_videos[:, cond_frames:]
    videos2 = result_videos[:, cond_frames:]
    ssim = calculate_ssim1(videos1, videos2)[0]
    psnr = calculate_psnr1(videos1, videos2)[0]
    lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
    
    print("[fvd    ]", fvd)
    print("[ssim   ]", ssim)
    print("[psnr   ]", psnr)
    print("[lpips  ]", lpips)

    return {
        'actual_step': actual_step,
        'metrics/fvd': fvd,
        'metrics/ssim': ssim,
        'metrics/psnr': psnr,
        'metrics/lpips': lpips
    }