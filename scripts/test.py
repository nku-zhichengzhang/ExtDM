import argparse

import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os

import yaml
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import timeit
import math
from utils.logger import Logger
from data.video_dataset import VideoDataset
import sys
import random
from einops import rearrange
from model.DM.video_flow_diffusion_model_pred_condframe_temp import FlowDiffusion

start = timeit.default_timer()
BATCH_SIZE = 8
VALID_BATCH_SIZE = 64
MAX_EPOCH = 4800
epoch_milestones = [2000, 3600]
root_dir = './logs_validation/diffusion/videoflowdiff_kth_test_0629'
data_dir = "/mnt/sda/hjy/kth/processed"
GPU = "0"
postfix = "-joint-steplr-random-onlyflow-train-regionmm-temp"  # sl: step-lr, rmm:regionmm
joint = "joint" in postfix or "-j" in postfix  # allow joint training with unconditional model
frame_sampling = "random" if "random" in postfix else "uniform"  # frame sampling strategy
only_use_flow = "onlyflow" in postfix or "-of" in postfix  # whether only use flow loss
if joint:
    null_cond_prob = 0.1
else:
    null_cond_prob = 0.0
split_train_test = "train" in postfix or "-tr" in postfix
use_residual_flow = "-rf" in postfix
config_pth = "./config/kth64.yaml"
# put your pretrained LFAE here
AE_RESTORE_FROM = "./logs_training/flow_pretrained/kth64/snapshots/RegionMM.pth"
RESTORE_FROM = "./logs_training/flowdiff.pth"
INPUT_SIZE = 64
CONDITION_FRAMES  = 10 # KTH
PRED_TRAIN_FRAMES = 20 # KTH
PRED_TEST_FRAMES  = 40 # KTH
VALID_VIDEO_NUM  =  256

LEARNING_RATE = 2e-4
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots'+postfix)
IMGSHOT_DIR = os.path.join(root_dir, 'imgshots'+postfix)
VIDSHOT_DIR = os.path.join(root_dir, "vidshots"+postfix)
SAMPLE_DIR = os.path.join(root_dir, 'sample'+postfix)
VALID_VIDEO_DIR = os.path.join(root_dir, 'valid_video_shots'+postfix)
NUM_EXAMPLES_PER_EPOCH = 479 # KTH
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
# SAVE_MODEL_EVERY = NUM_STEPS_PER_EPOCH * (MAX_EPOCH // 15)
# SAVE_VID_EVERY = 1000
# SAMPLE_VID_EVERY = 2000
# UPDATE_MODEL_EVERY = 3000

SAVE_VID_EVERY = 500
SAMPLE_VID_EVERY = 1000
UPDATE_MODEL_EVERY = 1000
SAVE_MODEL_EVERY = 1000

SAVE_IMG_EVERY = 100 # commented

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
print("image size, num frames:", INPUT_SIZE)
print("cond frames:", CONDITION_FRAMES  )
print("pred train frames:", PRED_TRAIN_FRAMES)
print("pred test frames:", PRED_TEST_FRAMES)
print("epoch milestones:", epoch_milestones)
print("split train test:", split_train_test)
print("frame sampling:", frame_sampling)
print("only use flow loss:", only_use_flow)
print("null_cond_prob:", null_cond_prob)
print("use residual flow:", use_residual_flow)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
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
    parser.add_argument('--save-img-freq', default=SAVE_IMG_EVERY, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument('--save-vid-freq', default=SAVE_VID_EVERY, type=int)
    parser.add_argument('--sample-vid-freq', default=SAMPLE_VID_EVERY, type=int)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--valid-batch-size", type=int, default=VALID_BATCH_SIZE, help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
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
    return parser.parse_args()


args = get_arguments()


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
        real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')
        
        origin_videos.append(real_vids)

        model.set_sample_input(cond_frame_num=CONDITION_FRAMES, tot_frame_num=CONDITION_FRAMES + PRED_TEST_FRAMES)
        pred_video = model.sample_one_video(cond_scale=1.0, real_vid=real_vids.cuda())['sample_out_vid'].clone().detach().cpu()

        result_videos.append(pred_video)

    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)

    origin_videos = rearrange(origin_videos, 'b c t h w -> b t c h w')
    result_videos = rearrange(result_videos, 'b c t h w -> b t c h w')
    from v import visualize
    visualize(
        save_path=VALID_VIDEO_DIR,
        origin=origin_videos,
        result=result_videos,
        epoch_num=epoch,
        save_pic_num=16,
        grid_nrow=4,
        save_pic_row=True,
        save_gif=False,
        cond_frame_num=CONDITION_FRAMES,
    )
    from fvd.calculate_fvd import calculate_fvd,calculate_fvd1
    from fvd.calculate_psnr import calculate_psnr,calculate_psnr1
    from fvd.calculate_ssim import calculate_ssim,calculate_ssim1
    from fvd.calculate_lpips import calculate_lpips,calculate_lpips1
    fvd = calculate_fvd1(origin_videos, result_videos, torch.device("cuda"), mini_bs=16)
    videos1 = origin_videos[:, CONDITION_FRAMES:CONDITION_FRAMES + PRED_TEST_FRAMES]
    videos2 = result_videos[:, CONDITION_FRAMES:CONDITION_FRAMES + PRED_TEST_FRAMES]
    ssim = calculate_ssim1(videos1, videos2)[0]
    psnr = calculate_psnr1(videos1, videos2)[0]
    lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
    
    print("[fvd    ]", fvd)
    print("[ssim   ]", ssim)
    print("[psnr   ]", psnr)
    print("[lpips  ]", lpips)
    return fvd, ssim, psnr, lpips



def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)
    
    with open(config_pth) as f:
        config = yaml.safe_load(f)

    model = FlowDiffusion(
                        is_train=True,
                        config=config,
                        pretrained_pth=AE_RESTORE_FROM,
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
                print(name)
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

    validloader = data.DataLoader( 
        VideoDataset(
            data_dir=data_dir, 
            type='valid', 
            image_size=args.input_size,
            num_frames=CONDITION_FRAMES + PRED_TEST_FRAMES,
            mean=MEAN, 
            total_videos=256
        ),
        batch_size=args.valid_batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    cnt = 0
    actual_step = args.start_step
    start_epoch = int(math.ceil((args.start_step * args.batch_size)/NUM_EXAMPLES_PER_EPOCH))
    epoch_cnt = start_epoch

    fvd, ssim, psnr, lpips = valid(validloader=validloader, model=model, epoch=epoch_cnt)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
