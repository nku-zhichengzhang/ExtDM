import numpy as np
import torch
from tqdm import tqdm
import math

import torch
import lpips

spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # value range [0, 1] -> [-1, 1]
    x = x * 2 - 1

    return x

def calculate_lpips(videos1, videos2, device):
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    # print("calculate_lpips...")

    assert videos1.shape == videos2.shape

    # videos [batch_size, timestamps, channel, h, w]

    # support grayscale input, if grayscale -> channel*3
    # value range [0, 1] -> [-1, 1]
    videos1 = trans(videos1)
    videos2 = trans(videos2)
    
    lpips_results = []

    for video_num in range(videos1.shape[0]):
    # for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] tensor

            img1 = video1[clip_timestamp].unsqueeze(0).to(device)
            img2 = video2[clip_timestamp].unsqueeze(0).to(device)
            
            loss_fn.to(device)

            # calculate lpips of a video
            lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
        lpips_results.append(lpips_results_of_a_video)
    
    lpips = {}
    lpips_std = {}

    for clip_timestamp in range(len(video1)):
        lpips[f'avg[{clip_timestamp}]'] = np.mean(lpips_results[:,clip_timestamp])
        lpips_std[f'std[{clip_timestamp}]'] = np.std(lpips_results[:,clip_timestamp])

    result = {
        "lpips": lpips,
        "lpips_std": lpips_std,
        "lpips_video_setting": video1.shape,
        "lpips_video_setting_name": "time, channel, heigth, width",
    }

    return result

def calculate_lpips1(videos1, videos2, device):
    assert videos1.shape == videos2.shape
    videos1 = trans(videos1)
    videos2 = trans(videos2)
    lpips_results = []
    for video_num in range(videos1.shape[0]):
    # for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].unsqueeze(0).to(device)
            img2 = video2[clip_timestamp].unsqueeze(0).to(device)
            loss_fn.to(device)
            lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
        lpips_results.append(lpips_results_of_a_video)
    lpips_results = np.array(lpips_results)
    return np.mean(lpips_results), np.std(lpips_results)

def calculate_lpips2(videos1, videos2, device):
    assert videos1.shape == videos2.shape
    videos1 = trans(videos1)
    videos2 = trans(videos2)
    lpips_results = []
    for video_num in range(videos1.shape[0]):
    # for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].unsqueeze(0).to(device)
            img2 = video2[clip_timestamp].unsqueeze(0).to(device)
            loss_fn.to(device)
            lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
        lpips_results.append(lpips_results_of_a_video)
    lpips_results = np.array(lpips_results)
    # print(np.mean(lpips_results,axis=-1))
    return np.min(np.mean(lpips_results,axis=-1))

def calculate_lpips3(videos1, videos2, device):
    assert videos1.shape == videos2.shape
    videos1 = trans(videos1)
    videos2 = trans(videos2)
    lpips_results = []
    for video_num in range(videos1.shape[0]):
    # for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].unsqueeze(0).to(device)
            img2 = video2[clip_timestamp].unsqueeze(0).to(device)
            loss_fn.to(device)
            lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
        lpips_results.append(lpips_results_of_a_video)
    lpips_results = np.array(lpips_results)
    # print(np.mean(lpips_results,axis=-1))
    return np.mean(lpips_results,axis=-1)

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 20
    CHANNEL = 3
    SIZE = 64
    CALCULATE_PER_FRAME = 5
    CALCULATE_FINAL = True
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    import json
    result = calculate_lpips2(videos1, videos2, device)
    # print(json.dumps(result, indent=4))
    print(result)

if __name__ == "__main__":
    main()