from metrics.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
import numpy as np
import torch
from tqdm import tqdm

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device):
    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = {}

    for clip_timestamp in tqdm(range(videos1.shape[-3])):

        # for calculate FVD, each clip_timestamp must >= 10
        if clip_timestamp < 10:
            continue

        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, : clip_timestamp]
        videos_clip2 = videos2[:, :, : clip_timestamp]

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
      
        # calculate FVD when timestamps[:clip]
        fvd_results[f'[{clip_timestamp}]'] = frechet_distance(feats1, feats2)

    result = {
        "fvd": fvd_results,
        "fvd_video_setting": videos1.shape,
        "fvd_video_setting_name": "batch_size, channel, time, heigth, width",
    }

    return result

def get_feats(videos, device, mini_bs=10):
    i3d = load_i3d_pretrained(device=device)
    videos = trans(videos)
    feats = get_fvd_feats(videos, i3d=i3d, device=device, bs=mini_bs)
    return feats

def calculate_fvd1(videos1, videos2, device, mini_bs=10):
    # assert videos1.shape == videos2.shape
    i3d = load_i3d_pretrained(device=device)
    videos1 = trans(videos1)
    videos2 = trans(videos2)
    feats1 = get_fvd_feats(videos1, i3d=i3d, device=device, bs=mini_bs)
    feats2 = get_fvd_feats(videos2, i3d=i3d, device=device, bs=mini_bs)
    return frechet_distance(feats1, feats2)

def calculate_fvd2(feats1, feats2):
    return frechet_distance(feats1, feats2)

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 100
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")
    mini_bs=2

    print(calculate_fvd1(videos1,videos2,device, mini_bs=16))

if __name__ == "__main__":
    main()