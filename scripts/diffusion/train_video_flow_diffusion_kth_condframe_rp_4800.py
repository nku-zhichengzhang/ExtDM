import argparse

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os

from utils.meter import AverageMeter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import timeit
import math
from PIL import Image
from utils.misc import Logger, grid2fig, conf2fig
from data.video_dataset import VideoDataset
import sys
import random
from einops import rearrange
from model.DM.video_flow_diffusion_model_pred_condframe_temp import FlowDiffusion
from torch.optim.lr_scheduler import MultiStepLR

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# FIXME: 更新为parser参数形式
start = timeit.default_timer()
BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
MAX_EPOCH = 4800
epoch_milestones = [3200, 4000]
root_dir = './logs_training/diffusion/videoflowdiff_kth_rf_temp'
data_dir = "/mnt/sda/hjy/kth/processed"
GPU = "1"
postfix = "-joint-steplr-random-onlyflow-train-regionmm-temp-rf"  # sl: step-lr, rmm:regionmm
joint = "joint" in postfix or "-j" in postfix  # allow joint training with unconditional model
frame_sampling = "random" if "random" in postfix else "uniform"  # frame sampling strategy
only_use_flow = "onlyflow" in postfix or "-of" in postfix  # whether only use flow loss
if joint:
    null_cond_prob = 0.1
else:
    null_cond_prob = 0.0
split_train_test = "train" in postfix or "-tr" in postfix
use_residual_flow = "-rf" in postfix
config_path = "./config/kth64.yaml"
# put your pretrained LFAE here
AE_RESTORE_FROM = "./logs_training/flow/kth64/snapshots/RegionMM_0100_S047900.pth"
INPUT_SIZE = 64
CONDITION_FRAMES = 10 # KTH
PRED_TEST_FRAMES = 10 # KTH
PRED_TRAIN_FRAMES = 10
VALID_VIDEO_NUM = 256

N_FRAMES = CONDITION_FRAMES + PRED_TEST_FRAMES

LEARNING_RATE = 2e-4
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
RESTORE_FROM = ""
SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots'+postfix)
IMGSHOT_DIR = os.path.join(root_dir, 'imgshots'+postfix)
VIDSHOT_DIR = os.path.join(root_dir, "vidshots"+postfix)
SAMPLE_DIR = os.path.join(root_dir, 'sample'+postfix)
NUM_EXAMPLES_PER_EPOCH = 479 # KTH
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
SAVE_MODEL_EVERY = NUM_STEPS_PER_EPOCH * (MAX_EPOCH // 15)
SAVE_VID_EVERY = 1000
SAMPLE_VID_EVERY = 2000
UPDATE_MODEL_EVERY = 3000

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(IMGSHOT_DIR, exist_ok=True)
os.makedirs(VIDSHOT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)
print(root_dir)
print("update saved model every:", UPDATE_MODEL_EVERY)
print("save model every:", SAVE_MODEL_EVERY)
print("save video every:", SAVE_VID_EVERY)
print("sample video every:", SAMPLE_VID_EVERY)
print(postfix)
print("RESTORE_FROM", RESTORE_FROM)
print("num examples per epoch:", NUM_EXAMPLES_PER_EPOCH)
print("max epoch:", MAX_EPOCH)
print("image size, num frames:", INPUT_SIZE, N_FRAMES)
print("epoch milestones:", epoch_milestones)
print("split train test:", split_train_test)
print("frame sampling:", frame_sampling)
print("only use flow loss:", only_use_flow)
print("null_cond_prob:", null_cond_prob)
print("use residual flow:", use_residual_flow)

def sample_img(rec_img_batch, idx=0):
    rec_img = rec_img_batch[idx].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array(MEAN)/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)

def valid(validloader, model, epoch):
    # b t c h w [0-1]
    origin_videos = []
    result_videos = []

    for i_iter, batch in enumerate(validloader):
        if i_iter >= VALID_VIDEO_NUM // VALID_BATCH_SIZE:
            break
        real_vids, real_names = batch
        origin_videos.append(real_vids)

        model.set_sample_input(cond_frame_num=CONDITION_FRAMES, tot_frame_num=CONDITION_FRAMES + PRED_TEST_FRAMES)
        pred_video = model.sample_one_video(cond_scale=1.0, real_vid=real_vids.cuda())['sample_out_vid'].clone().detach().cpu()
        # print(real_vids[:, :, :CONDITION_FRAMES, :, :].shape)

        res_video = torch.cat([real_vids[:, :, :CONDITION_FRAMES, :, :], pred_video], dim=2)
        result_videos.append(res_video)

    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)

    origin_videos = rearrange(origin_videos, 'b c t h w -> b t c h w')
    result_videos = rearrange(result_videos, 'b c t h w -> b t c h w')
    from utils.visualize import visualize
    visualize(
        save_path=root_dir,
        origin=origin_videos,
        result=result_videos,
        epoch_num=epoch,
        save_pic_num=8,
        grid_nrow=4,
        save_pic_row=False,
        save_gif=False,
        cond_frame_num=10,
    )
    from metrics.calculate_fvd import calculate_fvd,calculate_fvd1
    from metrics.calculate_psnr import calculate_psnr,calculate_psnr1
    from metrics.calculate_ssim import calculate_ssim,calculate_ssim1
    from metrics.calculate_lpips import calculate_lpips,calculate_lpips1
    fvd = calculate_fvd1(origin_videos, result_videos, torch.device("cuda"), mini_bs=2)
    videos1 = origin_videos[:, CONDITION_FRAMES:CONDITION_FRAMES + PRED_TEST_FRAMES]
    videos2 = result_videos[:, CONDITION_FRAMES:CONDITION_FRAMES + PRED_TEST_FRAMES]
    ssim = calculate_ssim1(videos1, videos2)[0]
    psnr = calculate_psnr1(videos1, videos2)[0]
    lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
    
    print("[fvd    ]", fvd)
    print("[ssim   ]", ssim)
    print("[psnr   ]", psnr)
    print("[lpips  ]", lpips)
    return fvd

if __name__ == '__main__':
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    cudnn.enabled = True
    cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--set-start", default=True)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--img-dir", type=str, default=IMGSHOT_DIR,
                        help="Where to save images of the model.")
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=100, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument('--save-vid-freq', default=SAVE_VID_EVERY, type=int)
    parser.add_argument('--sample-vid-freq', default=SAMPLE_VID_EVERY, type=int)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--valid-batch-size", type=int, default=VALID_BATCH_SIZE, help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--n-frames", default=N_FRAMES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_MODEL_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--update-pred-every", type=int, default=UPDATE_MODEL_EVERY)
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--fp16", default=False)
    
    args = parser.parse_args()

    setup_seed(args.random_seed)

    model = FlowDiffusion(is_train=True,
                          img_size=INPUT_SIZE//2,
                          num_frames=CONDITION_FRAMES + PRED_TRAIN_FRAMES,
                          null_cond_prob=null_cond_prob,
                          sampling_timesteps=10,
                          only_use_flow=only_use_flow,
                          use_residual_flow=use_residual_flow,
                          config_path=config_path,
                          pretrained_pth=AE_RESTORE_FROM,
                          cond_num=CONDITION_FRAMES,
                          pred_num=PRED_TRAIN_FRAMES,
                          )

    model.cuda()
    optimizer_diff = torch.optim.Adam(model.diffusion.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))
    # Not set model to be train mode! Because pretrained flow autoenc need to be eval

    if args.fine_tune:
        pass
    elif args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
            model_ckpt = model.diffusion.state_dict()
            for name, _ in model_ckpt.items():
                model_ckpt[name].copy_(checkpoint['diffusion'][name])
            model.diffusion.load_state_dict(model_ckpt)
            print("=> loaded checkpoint '{}'".format(args.restore_from))
            if "optimizer_diff" in list(checkpoint.keys()):
                optimizer_diff.load_state_dict(checkpoint['optimizer_diff'])
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
    else:
        print("NO checkpoint found!")
    
    setup_seed(args.random_seed)

    trainloader = data.DataLoader(VideoDataset(
                                    data_dir=data_dir,
                                    type='train', 
                                    image_size=INPUT_SIZE,
                                    num_frames=CONDITION_FRAMES+PRED_TRAIN_FRAMES,
                                    mean=MEAN
                                ),
                                batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)
    validloader = data.DataLoader(VideoDataset(
                                    data_dir=data_dir,
                                    type='valid', 
                                    image_size=INPUT_SIZE,
                                    num_frames=N_FRAMES,
                                    mean=MEAN,
                                    total_videos=256
                                ),
                                batch_size=args.valid_batch_size,
                                shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_rec = AverageMeter()
    losses_warp = AverageMeter()

    cnt = 0
    actual_step = args.start_step
    start_epoch = int(math.ceil((args.start_step * args.batch_size)/NUM_EXAMPLES_PER_EPOCH))
    epoch_cnt = start_epoch

    scheduler = MultiStepLR(optimizer_diff, epoch_milestones, gamma=0.1, last_epoch=start_epoch - 1)
    print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer_diff.param_groups[0]["lr"]))
    fvd_best = 1e5
    while actual_step < args.final_step:
        iter_end = timeit.default_timer()

        for i_iter, batch in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)

            real_vids, real_names = batch
            # use first frame of each video as reference frame

            real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')

            print(real_vids.shape)
            # torch.Size([bs, length, c, h, w])
            ref_imgs = real_vids[:, :, CONDITION_FRAMES-1, :, :].clone().detach()
            bs = real_vids.size(0)

            model.set_train_input(cond_frame_num=CONDITION_FRAMES, train_frame_num=PRED_TRAIN_FRAMES, tot_frame_num=CONDITION_FRAMES + PRED_TRAIN_FRAMES)
            ret = model.optimize_parameters(real_vids[:,:,:CONDITION_FRAMES + PRED_TRAIN_FRAMES].cuda(), optimizer_diff)

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            losses.update(ret['loss'], bs)
            losses_rec.update(ret['rec_loss'], bs)
            losses_warp.update(ret['rec_warp_loss'], bs)

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.7f} ({loss.avg:.7f})\t'
                      'loss_rec {loss_rec.val:.4f} ({loss_rec.avg:.4f})\t'
                      'loss_warp {loss_warp.val:.4f} ({loss_warp.avg:.4f})'
                    .format(
                    cnt, actual_step, args.final_step,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_rec=losses_rec,
                    loss_warp=losses_warp,
                ))

            if actual_step % args.save_img_freq == 0:
                msk_size = ref_imgs.shape[-1]
                save_src_img = sample_img(ref_imgs)
                save_tar_img = sample_img(real_vids[:, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2, :, :])
                save_real_out_img = sample_img(ret['real_out_vid'][:, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2, :, :])
                save_real_warp_img = sample_img(ret['real_warped_vid'][:, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2, :, :])
                save_fake_out_img = sample_img(ret['fake_out_vid'][:, :, PRED_TRAIN_FRAMES//2, :, :])
                save_fake_warp_img = sample_img(ret['fake_warped_vid'][:, :, PRED_TRAIN_FRAMES//2, :, :])
                save_real_grid = grid2fig(ret['real_vid_grid'][0, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=32, img_size=msk_size)
                save_fake_grid = grid2fig(ret['fake_vid_grid'][0, :, PRED_TRAIN_FRAMES//2].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=32, img_size=msk_size)
                save_real_conf = conf2fig(ret['real_vid_conf'][0, :, CONDITION_FRAMES+PRED_TRAIN_FRAMES//2], img_size=INPUT_SIZE)
                save_fake_conf = conf2fig(ret['fake_vid_conf'][0, :, PRED_TRAIN_FRAMES//2], img_size=INPUT_SIZE)
                new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                new_im.paste(Image.fromarray(save_fake_out_img, 'RGB'), (msk_size * 2, 0))
                new_im.paste(Image.fromarray(save_fake_warp_img, 'RGB'), (msk_size * 2, msk_size))
                new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                new_im_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                              + '_' + format(real_names[0], "06d") + ".png"
                new_im_file = os.path.join(args.img_dir, new_im_name)
                new_im.save(new_im_file)

            if actual_step % args.save_vid_freq == 0:
                print("saving video...")
                # num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                save_src_img = sample_img(ref_imgs)
                for nf in range(CONDITION_FRAMES + PRED_TRAIN_FRAMES):
                    save_tar_img = sample_img(real_vids[:, :, nf, :, :])
                    save_real_out_img = sample_img(ret['real_out_vid'][:, :, nf, :, :])
                    save_real_warp_img = sample_img(ret['real_warped_vid'][:, :, nf, :, :])
                    save_fake_out_img = sample_img(ret['fake_out_vid'][:, :, nf - CONDITION_FRAMES, :, :])
                    save_fake_warp_img = sample_img(ret['fake_warped_vid'][:, :, nf - CONDITION_FRAMES, :, :])
                    save_real_grid = grid2fig(
                        ret['real_vid_grid'][0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_fake_grid = grid2fig(
                        ret['fake_vid_grid'][0, :, nf - CONDITION_FRAMES].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_real_conf = conf2fig(ret['real_vid_conf'][0, :, nf], img_size=INPUT_SIZE)
                    save_fake_conf = conf2fig(ret['fake_vid_conf'][0, :, nf - CONDITION_FRAMES], img_size=INPUT_SIZE)
                    new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                    new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                    new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                    new_im.paste(Image.fromarray(save_fake_out_img, 'RGB'), (msk_size * 2, 0))
                    new_im.paste(Image.fromarray(save_fake_warp_img, 'RGB'), (msk_size * 2, msk_size))
                    new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                    new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                    new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                    new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                                + '_' + format(real_names[0], "06d") + ".gif"
                new_vid_file = os.path.join(VIDSHOT_DIR, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)

            # sampling
            if actual_step % args.sample_vid_freq == 0:#  and cnt != 0
                print("sampling video...")
                model.set_sample_input(cond_frame_num=CONDITION_FRAMES, tot_frame_num=CONDITION_FRAMES + PRED_TEST_FRAMES)
                ret = model.sample_one_video(cond_scale=1.0, real_vid=real_vids.cuda())
                # num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                save_src_img = sample_img(ref_imgs)
                for nf in range(CONDITION_FRAMES + PRED_TRAIN_FRAMES):
                    save_tar_img = sample_img(real_vids[:, :, nf , :, :])
                    save_real_out_img = sample_img(ret['real_out_vid'][:, :, nf, :, :])
                    save_real_warp_img = sample_img(ret['real_warped_vid'][:, :, nf, :, :])
                    save_sample_out_img = sample_img(ret['sample_out_vid'][:, :, nf, :, :])
                    save_sample_warp_img = sample_img(ret['sample_warped_vid'][:, :, nf, :, :])
                    save_real_grid = grid2fig(
                        ret['real_vid_grid'][0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_fake_grid = grid2fig(
                        ret['sample_vid_grid'][0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_real_conf = conf2fig(ret['real_vid_conf'][0, :, nf], img_size=INPUT_SIZE)
                    save_fake_conf = conf2fig(ret['sample_vid_conf'][0, :, nf], img_size=INPUT_SIZE)
                    new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                    new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                    new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                    new_im.paste(Image.fromarray(save_sample_out_img, 'RGB'), (msk_size * 2, 0))
                    new_im.paste(Image.fromarray(save_sample_warp_img, 'RGB'), (msk_size * 2, msk_size))
                    new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                    new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                    new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                    new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                                + '_' + format(real_names[0], "06d") + ".gif"
                new_vid_file = os.path.join(SAMPLE_DIR, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)
            # if actual_step % args.sample_vid_freq == 0:

            # save model at i-th step
            if actual_step % args.save_pred_every == 0 and cnt != 0:
                # run validation
                fvd = valid(validloader=validloader, model=model, epoch=epoch_cnt)
                if fvd<fvd_best:
                    fvd_best = fvd
                    epoch_best = epoch_cnt
                print('=====================================')
                print('history best fvd:', fvd_best)
                print('history best epoch:', epoch_best)
                print('=====================================')
                # save model
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'diffusion': model.diffusion.state_dict(),
                            'optimizer_diff': optimizer_diff.state_dict()},
                           osp.join(args.snapshot_dir,
                                    'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

            # update saved model
            if actual_step % args.update_pred_every == 0 and cnt != 0:
                valid(validloader=validloader, model=model, epoch=epoch_cnt)
                print('updating saved snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'diffusion': model.diffusion.state_dict(),
                            'optimizer_diff': optimizer_diff.state_dict()},
                           osp.join(args.snapshot_dir, 'flowdiff.pth'))

            if actual_step >= args.final_step:
                break

            cnt += 1

        scheduler.step()
        epoch_cnt += 1
        print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer_diff.param_groups[0]["lr"]))

    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'diffusion': model.diffusion.state_dict(),
                'optimizer_diff': optimizer_diff.state_dict()},
               osp.join(args.snapshot_dir,
                        'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))
    end = timeit.default_timer()
    print(end - start, 'seconds')
