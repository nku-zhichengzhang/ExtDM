import cv2, shutil
import os, torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from natsort import natsorted
from metrics.calculate_fvd import calculate_fvd,calculate_fvd1
from metrics.calculate_psnr import calculate_psnr,calculate_psnr1
from metrics.calculate_ssim import calculate_ssim,calculate_ssim1
from metrics.calculate_lpips import calculate_lpips,calculate_lpips1
    
def get_metrics(videos1, videos2):
    videos1 = np.array(videos1).astype(float)/255
    videos1 = torch.from_numpy(videos1)
    videos1 = rearrange(videos1, 't h w c-> 1 t c h w').float()

    videos2 = np.array(videos2).astype(float)/255
    videos2 = torch.from_numpy(videos2)
    videos2 = rearrange(videos2, 't h w c-> 1 t c h w').float()
    # fvd = calculate_fvd1(videos1, videos2, torch.device("cuda"), mini_bs=1)
    ssim = calculate_ssim1(videos1, videos2)[0]
    psnr = calculate_psnr1(videos1, videos2)[0]
    lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
    return psnr, ssim, lpips

dataset = '/home/ubuntu/zzc/data/video_prediction/ExtDM_output/SMMNIST/smmnist64_DM_Batch32_lr2.0e-4_c5p5_STW_adaptor_multi_124_resume_flowdiff_0036_S265000_1000_100'
samplenum=100
c=10
p=10
saveroot = '/home/ubuntu/zzc/data/video_prediction/tile'
savedata = 'SMMNIST'

if os.path.exists(os.path.join(saveroot, savedata)):
    shutil.rmtree(os.path.join(saveroot, savedata))
os.makedirs(os.path.join(saveroot, savedata))

origin_fold = os.path.join(dataset, 'result_origin', '0', 'pic')

for vid in tqdm(os.listdir(origin_fold)):
    gts = []
    gt_vid = os.path.join(origin_fold, vid, 'result')
    ori_p = natsorted(os.listdir(gt_vid))

    for i in range(c+p):
        gts.append(cv2.imread(os.path.join(gt_vid, ori_p[i])))
    gt_imgs = np.concatenate(gts, axis=1)

    psnr_best = None
    ssim_best = None
    lpips_best = None
    fvd_best = None

    psnr,ssim = 0,0
    lpips,fvd = 1e8,1e8

    
    for sample in range(samplenum):
        # print(sample)
        ress = []
        sample_vid = os.path.join(dataset, 'result_'+str(sample), '0', 'pic', vid, 'result')
        res_p = natsorted(os.listdir(sample_vid))

        for i in range(c):
            ress.append(np.zeros_like(gts[-1]))
        for i in range(p):
            ress.append(cv2.imread(os.path.join(sample_vid, res_p[c+i])))
            # print(os.path.join(sample_vid, res_p[i]))

        res_imgs = np.concatenate(ress, axis=1)

        psnr_cur, ssim_cur, lpips_cur = get_metrics(gts[c:], ress[c:])
        # print(psnr_cur, 'psnr_cur')
        if psnr_cur > psnr:
            psnr = psnr_cur
            # print(psnr, 'psnr')
            psnr_best = res_imgs
        if ssim_cur > ssim:
            ssim = ssim_cur
            # print(ssim, 'ssim')
            ssim_best = res_imgs
        if lpips_cur < lpips:
            lpips = lpips_cur
            # print(lpips, 'lpips')
            lpips_best = res_imgs



    psnrimg = np.concatenate([gt_imgs, psnr_best], axis=0)
    cv2.imwrite(os.path.join(saveroot, savedata, vid + '_psnr' + '.png'), psnrimg)

    ssimimg = np.concatenate([gt_imgs, ssim_best], axis=0)
    cv2.imwrite(os.path.join(saveroot, savedata, vid + '_ssim' + '.png'), ssimimg)

    lpipsimg = np.concatenate([gt_imgs, lpips_best], axis=0)
    cv2.imwrite(os.path.join(saveroot, savedata, vid + '_lpips' + '.png'), lpipsimg)

    # fvdimg = np.concatenate([gt_imgs, fvd_best], axis=0)
    # cv2.imwrite(os.path.join(saveroot, savedata, vid + '_fvd' + '.png'), fvdimg)
    