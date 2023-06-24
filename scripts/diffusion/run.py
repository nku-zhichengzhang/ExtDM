import argparse

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import yaml
from shutil import copy

from utils.meter import AverageMeter
from utils.misc import grid2fig, conf2fig
from utils.seed import setup_seed
from utils.logger import Logger
from data.video_dataset import VideoDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os.path as osp
import timeit
import math
from PIL import Image
import sys
import random
from einops import rearrange
from model.DM.video_flow_diffusion_model_pred_condframe_temp import FlowDiffusion
from torch.optim.lr_scheduler import MultiStepLR

# # FIXME: 更新为parser参数形式
# start = timeit.default_timer()
# BATCH_SIZE = 8
# VALID_BATCH_SIZE = 8
# MAX_EPOCH = 4800
# epoch_milestones = [3200, 4000]
# root_dir = './logs_training/diffusion/kth0621'
# data_dir = "/mnt/sda/hjy/kth/processed"
# GPU = "1"
postfix = "-joint-steplr-random-onlyflow-train-regionmm-temp-rf"  # sl: step-lr, rmm:regionmm
# frame_sampling = "random" if "random" in postfix else "uniform"  # frame sampling strategy
# only_use_flow = "onlyflow" in postfix or "-of" in postfix  # whether only use flow loss
# split_train_test = "train" in postfix or "-tr" in postfix

# # put your pretrained LFAE here
# INPUT_SIZE = 64
# CONDITION_FRAMES = 10 # KTH
# PRED_TEST_FRAMES = 10 # KTH
# PRED_TRAIN_FRAMES = 10
# VALID_VIDEO_NUM = 256

# N_FRAMES = CONDITION_FRAMES + PRED_TEST_FRAMES

# LEARNING_RATE = 2e-4
# RANDOM_SEED = 1234
# MEAN = (0.0, 0.0, 0.0)
# RESTORE_FROM = ""
# SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots'+postfix)
# IMGSHOT_DIR = os.path.join(root_dir, 'imgshots'+postfix)
# VIDSHOT_DIR = os.path.join(root_dir, "vidshots"+postfix)
# SAMPLE_DIR = os.path.join(root_dir, 'sample'+postfix)
# NUM_EXAMPLES_PER_EPOCH = 479 # KTH
# NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
# MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
#                NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
# SAVE_MODEL_EVERY = NUM_STEPS_PER_EPOCH * (MAX_EPOCH // 15)
# SAVE_VID_EVERY = 1000
# SAMPLE_VID_EVERY = 2000
# UPDATE_MODEL_EVERY = 3000

# os.makedirs(SNAPSHOT_DIR, exist_ok=True)
# os.makedirs(IMGSHOT_DIR, exist_ok=True)
# os.makedirs(VIDSHOT_DIR, exist_ok=True)
# os.makedirs(SAMPLE_DIR, exist_ok=True)


# print(root_dir)
# print("update saved model every:", UPDATE_MODEL_EVERY)
# print("save model every:", SAVE_MODEL_EVERY)
# print("save video every:", SAVE_VID_EVERY)
# print("sample video every:", SAMPLE_VID_EVERY)
# print(postfix)
# print("RESTORE_FROM", RESTORE_FROM)
# print("num examples per epoch:", NUM_EXAMPLES_PER_EPOCH)
# print("max epoch:", MAX_EPOCH)
# print("image size, num frames:", INPUT_SIZE, N_FRAMES)
# print("epoch milestones:", epoch_milestones)
# print("split train test:", split_train_test)
# print("frame sampling:", frame_sampling)
# print("only use flow loss:", only_use_flow)
# print("null_cond_prob:", null_cond_prob)
# print("use residual flow:", use_residual_flow)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    cudnn.enabled = True
    cudnn.benchmark = True

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--postfix", default="")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--set-start", default=True)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--log_dir",default='./logs_training/diffusion', help="path to log into")
    parser.add_argument("--config",default="./config/kth64.yaml",help="path to config")
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--device_ids", default="0,1", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default="")
    parser.add_argument("--checkpoint", # use the pretrained model provided by Snap
                        default="./logs_training/flow/kth64_test/snapshots/RegionMM.pth",
                        help="path to checkpoint")
    parser.add_argument("--verbose", default=False, help="Print model architecture")

    # parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
    #                     help="Number of training steps.")
    # parser.add_argument('--print-freq', '-p', default=10, type=int,
    #                     metavar='N', help='print frequency')
    # parser.add_argument('--save-img-freq', default=100, type=int,
    #                     metavar='N', help='save image frequency')
    # parser.add_argument('--save-vid-freq', default=SAVE_VID_EVERY, type=int)
    # parser.add_argument('--sample-vid-freq', default=SAMPLE_VID_EVERY, type=int)
    # parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
    #                     help="Number of images sent to the network in one step.")
    # parser.add_argument("--valid-batch-size", type=int, default=VALID_BATCH_SIZE, help="Number of images sent to the network in one step.")
    # parser.add_argument("--input-size", 
    #                     type=int, 
    #                     default=128,
    #                     help="height and width of videos")
    # parser.add_argument("--n-frames", default=N_FRAMES)
    # parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
    #                     help="Base learning rate for training with polynomial decay.")
    # parser.add_argument("--save-pred-every", type=int, default=SAVE_MODEL_EVERY,
    #                     help="Save checkpoint every often.")
    # parser.add_argument("--update-pred-every", type=int, default=UPDATE_MODEL_EVERY)

    # parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
    #                     help="Where to save snapshots of the model.")
    
    parser.add_argument("--fp16", default=False)
    
    args = parser.parse_args()

    setup_seed(args.random_seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.postfix == '':
        postfix = ''
    else:
        postfix = '_' + args.postfix

    log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0]+postfix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        copy(args.config, log_dir)

    # the directory to save checkpoints
    config["snapshots"] = os.path.join(log_dir, 'snapshots')
    os.makedirs(config["snapshots"], exist_ok=True)
    # the directory to save images of training results
    config["imgshots"] = os.path.join(log_dir, 'imgshots')
    os.makedirs(config["imgshots"], exist_ok=True)

    config["set_start"] = args.set_start

    train_params   = config['diffusion_params']['train_params']
    model_params   = config['diffusion_params']['model_params']
    dataset_params = config['dataset_params'] 

    log_txt = os.path.join(log_dir,
                           "B"+format(train_params['batch_size'], "04d")+
                           "E"+format(train_params['max_epochs'], "04d")+".log")
    sys.stdout = Logger(log_txt, sys.stdout)

    wandb.login()
    wandb.init(
        project="EDM_v1",
        config={
            "learning_rate": train_params['lr'],
            "epochs": train_params['max_epochs'],
        },
        name=f"{config['experiment_name']}{postfix}",
        dir=log_dir
    )

    print("postfix:", postfix)
    print("checkpoint:", args.checkpoint)
    print("batch size:", train_params['batch_size'])

    if torch.cuda.is_available():
        diffusion.to(args.device_ids[0])
    if args.verbose:
        print(diffusion)
    # Not set model to be train mode! Because pretrained flow autoenc need to be eval

    print("Training...")
    train(
        config, 
        dataset_params,
        train_params, 
        log_dir, 
        args.checkpoint, 
        args.device_ids
    )
    