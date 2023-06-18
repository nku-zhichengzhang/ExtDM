import torch
from metrics.calculate_fvd import calculate_fvd
from metrics.calculate_psnr import calculate_psnr
from metrics.calculate_ssim import calculate_ssim
from metrics.calculate_lpips import calculate_lpips

# ps: pixel value should be in [0, 1]!

NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 30
CHANNEL = 3
SIZE = 64
CALCULATE_PER_FRAME = 8
CALCULATE_FINAL = True
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda")
# device = torch.device("cpu")

import json
result = {}
result['fvd'] = calculate_fvd(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
result['ssim'] = calculate_ssim(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL)
result['psnr'] = calculate_psnr(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL)
result['lpips'] = calculate_lpips(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
print(json.dumps(result, indent=4))
