import einops
import os
import mediapy as media
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from utils.misc import grid2fig
from PIL import Image
import imageio

def visualize(save_path, origin, result, epoch_or_step_num=0, cond_frame_num=10, skip_pic_num=1, save_pic_num=8, grid_nrow=4, save_pic_row=True, save_gif=True, save_gif_grid=True):
    # 输入: origin [b t c h w] + result [b t c h w]
    
    # 数据判定
    print("save", save_pic_num, "samples")
    assert origin.shape == result.shape, f"origin ({origin.shape}) result ({result.shape}) shape are not equal."
    assert cond_frame_num <= origin.shape[1], f"cond_frame_num ({cond_frame_num}) is too big for video length ({origin.shape[1]})."
    
    # 确认文件夹存在
    epoch_or_step_save_path = os.path.join(save_path, f"{epoch_or_step_num}")
    os.makedirs(epoch_or_step_save_path, exist_ok=True)

    # 确认输出图像数量
    save_pic_num = min (len(origin), save_pic_num)
    if len(origin) < save_pic_num:
        save_pic_num = min (len(origin), save_pic_num)
        print(f"video batchsize({len(origin)}) is too small, save_num is set to {save_pic_num}")
    
    index = [int(i) for i in torch.linspace(0, len(origin)-1, save_pic_num)]
    
    print(index)

    origin = origin[index].cpu()
    result = result[index].cpu()

    # 输出视频分解对比图
    if save_pic_row:
        save_pic_row_path = os.path.join(epoch_or_step_save_path, "pic_row")
        os.makedirs(save_pic_row_path, exist_ok=True)
        
        all_video = torch.stack([origin, result])
        
        for i in range(save_pic_num):
            two_video = all_video[:, i]
            two_video[1, :cond_frame_num] = 1.0

            two_video = two_video[:,::skip_pic_num]

            # # two_video       [2 t c h w]
            # # two_video_cond  [2 :cond c h w]
            # # two_video_pred  [2 cond::skip t c h w]
            # # print(two_video.shape)

            # two_video_cond           = two_video[:,:cond_frame_num]
            # two_video_pred_with_skip = two_video[:,cond_frame_num::skip_pic_num]

            # two_video_with_skip = torch.cat([two_video_cond, two_video_pred_with_skip], dim=1)

            # # print(two_video_with_skip.shape)



            # result of cond_frame set to blank frame
            two_video = einops.rearrange(two_video, "b t c h w -> (b h) (t w) c")
            save_path = os.path.join(save_pic_row_path, f"pic_row_{epoch_or_step_num}_sample{i}.png")
            # print(save_path)
            media.write_image(save_path, two_video.squeeze().numpy())

    # 输出视频
    if save_gif:
        save_gif_path = os.path.join(epoch_or_step_save_path, "gif")
        os.makedirs(save_gif_path, exist_ok=True)
        
        origin_output = einops.rearrange(origin, "b t c h w -> b t h w c")
        result_output = einops.rearrange(result, "b t c h w -> b t h w c")

        for i in range(save_pic_num):
            media.write_video(os.path.join(save_gif_path, f"gif_{epoch_or_step_num}_origin{i}.gif"), origin_output[i].squeeze().numpy(), codec='gif', fps=20)
            media.write_video(os.path.join(save_gif_path, f"gif_{epoch_or_step_num}_result{i}.gif"), result_output[i].squeeze().numpy(), codec='gif', fps=20)

    # 输出视频对比网格
    if save_gif_grid:
        save_gif_grid_path = os.path.join(epoch_or_step_save_path, "gif_grid")
        os.makedirs(save_gif_grid_path, exist_ok=True)

        cond_color = [3/255,87/255,127/255]
        pred_color = [254/255,85/255,1/255]

        if origin.shape[2] == 1:
            origin_output = einops.repeat(origin, "b t c h w -> b t (3 c) h w")
            result_output = einops.repeat(result, "b t c h w -> b t (3 c) h w")
            origin_output = einops.rearrange(origin_output, "b t c h w -> b t h w c").numpy()
            result_output = einops.rearrange(result_output, "b t c h w -> b t h w c").numpy()
        else:
            origin_output = einops.rearrange(origin, "b t c h w -> b t h w c").numpy()
            result_output = einops.rearrange(result, "b t c h w -> b t h w c").numpy()

        videos_grids = []

        for i in range(save_pic_num):
            origin_video = origin_output[i]
            result_video = result_output[i]

            video_grids=[]

            for t in range(len(origin_video)):
                
                if t < cond_frame_num:
                    origin_img = cv2.copyMakeBorder(origin_video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=cond_color)
                    result_img = cv2.copyMakeBorder(result_video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=cond_color)
                else:
                    origin_img = cv2.copyMakeBorder(origin_video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=cond_color)
                    result_img = cv2.copyMakeBorder(result_video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=pred_color)

                video_grid = np.stack([origin_img, result_img])
                video_grid = einops.rearrange(video_grid, "n h w c -> h (n w) c")

                video_grids.append(video_grid)
                
            video_grids = np.stack(video_grids)

            videos_grids.append(video_grids)

        videos_grids = np.stack(videos_grids)

        videos_grids = torch.from_numpy(videos_grids)
        videos_grids = einops.rearrange(videos_grids, "b t h w c-> t b c h w")


        final_grids = []

        for t in range(len(videos_grids)):
            # make_grid need [b c h w]
            final_grids.append(make_grid(videos_grids[t], nrow=grid_nrow, padding=10, pad_value=1))

        final_grids = torch.stack(final_grids)
        final_grids = einops.rearrange(final_grids, "t c h w -> t h w c")

        media.write_video(os.path.join(save_gif_grid_path, f"save_gif_grid_sample.gif"), final_grids.numpy(), codec='gif', fps=20)



def LFAE_visualize(
            ground, prediction, deformed, optical_flow, occlusion_map, video_names,
            save_path, save_num=8, epoch_or_step_num=0, image_size=64
        ):

    # 确认文件夹存在
    epoch_or_step_save_path = os.path.join(save_path, f"{epoch_or_step_num}")
    os.makedirs(epoch_or_step_save_path, exist_ok=True)

    index = [int(i) for i in torch.linspace(0, ground.size(0)-1, save_num)]
    
    for batch_idx in index:
        new_im_list = []
        for frame_idx in range(ground.size(2)):
            # cond+real
            save_tar_img = sample_img(ground[:, :, frame_idx], batch_idx)
            # prediction
            save_out_img = sample_img(prediction[frame_idx], batch_idx)
            # deformed
            save_warped_img = sample_img(deformed[frame_idx], batch_idx)
            # optical_flow
            save_warped_grid = grid2fig(optical_flow[frame_idx, batch_idx].data.cpu().numpy(),grid_size=32, img_size=image_size)
            # occlusion_map
            save_conf_map = occlusion_map[frame_idx, batch_idx].unsqueeze(dim=0)
            save_conf_map = save_conf_map.data.cpu()
            save_conf_map = F.interpolate(save_conf_map, size=ground.shape[3:5]).numpy()
            save_conf_map = np.transpose(save_conf_map, [0, 2, 3, 1])
            save_conf_map = np.array(save_conf_map[0, :, :, 0]*255, dtype=np.uint8)
            # save img_list
            new_im = Image.new('RGB', (image_size * 5, image_size))
            new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
            new_im.paste(Image.fromarray(save_out_img, 'RGB'), (image_size, 0))
            new_im.paste(Image.fromarray(save_warped_img, 'RGB'), (image_size * 2, 0))
            new_im.paste(Image.fromarray(save_warped_grid), (image_size * 3, 0))
            new_im.paste(Image.fromarray(save_conf_map, "L"), (image_size * 4, 0))
            new_im_list.append(new_im)
        imageio.mimsave(os.path.join(epoch_or_step_save_path, f"{video_names[batch_idx]}.gif"), new_im_list)

def sample_img(rec_img_batch, index):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array((0.0, 0.0, 0.0))/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)