# loading video dataset for training and testing
import os
import torch

import numpy as np
import torch.utils.data as data
# from skimage.color import gray2rgb

import cv2
# import torchvision.transforms.functional as F
from torchvision import transforms

from data.h5 import HDF5Dataset

from einops import rearrange, repeat

def dataset2video(video):
    if len(video.shape) == 3:
        video = repeat(video, 't h w -> t c h w', c=3)
    elif video.shape[1] == 1:
        video = repeat(video, 't c h w -> t (n c) h w', n=3)
    else:
        video = rearrange(video, 't h w c -> t c h w')
    return video

def dataset2videos(videos):
    if len(videos.shape) == 4:
        videos = repeat(videos, 'b t h w -> b t c h w', c=3)
    elif videos.shape[2] == 1:
        videos = repeat(videos, 'b t c h w -> b t (n c) h w', n=3)
    else:
        videos = rearrange(videos, 'b t h w c -> b t c h w')
    return videos
        
def resize(im, desired_size, interpolation):
    old_size = im.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple(int(x*ratio) for x in old_size)

    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=interpolation)
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im

class VideoDataset(data.Dataset):
    def __init__(self, 
        data_dir, 
        type='train',
        total_videos=-1, 
        num_frames=40, 
        image_size=64, 
        random_time=True,
        # color_jitter=None, 
        random_horizontal_flip=False
    ):
        super(VideoDataset, self).__init__()
        self.data_dir = data_dir
        self.type = type
        self.num_frames = num_frames
        self.image_size = image_size
        self.total_videos = total_videos
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        # self.jitter = transforms.ColorJitter(hue=color_jitter) if color_jitter else None
        
        if "UCF" in self.data_dir:
            self.videos_ds = HDF5Dataset(self.data_dir)
            # Train
            # self.num_train_vids = 9624
            # self.num_test_vids = 3696   # -> 369 : https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
            with self.videos_ds.opener(self.videos_ds.shard_paths[0]) as f:
                self.num_train_vids = f['num_train'][()]
                self.num_test_vids = f['num_test'][()]//10  # https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
        else:
            self.videos_ds = HDF5Dataset(os.path.join(self.data_dir, type))
        
    def __len__(self):
        if self.total_videos > 0:
            return self.total_videos
        else:
            if "UCF" in self.data_dir:
                return self.num_train_vids if self.type=='train' else self.num_test_vids
            else:
                return len(self.videos_ds)

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def max_index(self):
        if "UCF" in self.data_dir:
            return self.num_train_vids if self.type=='train' else self.num_test_vids
        else:  
            return len(self.videos_ds)

    def __getitem__(self, index, time_idx=0):
        if "UCF" in self.data_dir:
            video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
            if not self.type=='train':
                video_index = video_index * 10 + self.num_train_vids    # https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
            shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

            # random crop
            crop_c = np.random.randint(int(self.image_size/240*320) - self.image_size) if self.type=='train' else int((self.image_size/240*320 - self.image_size)/2)

            # random horizontal flip
            flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0

            # read data
            prefinals = []
            with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
                total_num_frames = f['len'][str(idx_in_shard)][()]

                # sample frames
                if self.random_time and total_num_frames > self.num_frames:
                    # sampling start frames
                    time_idx = np.random.choice(total_num_frames - self.num_frames)
                # read frames
                for i in range(time_idx, min(time_idx + self.num_frames, total_num_frames)):
                    img = f[str(idx_in_shard)][str(i)][()]
                    arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img[:, crop_c:crop_c + self.image_size]))
                    prefinals.append(arr)

            data = torch.stack(prefinals)
            data = rearrange(data, "t c h w -> t h w c")
            return data, video_index

        else:
            video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
            shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
            
            prefinals = []
            with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
                total_num_frames = f['len'][str(idx_in_shard)][()]

                # sample frames
                if self.random_time and total_num_frames > self.num_frames:
                    # sampling start frames
                    time_idx = np.random.choice(total_num_frames - self.num_frames)
                # read frames
                for i in range(time_idx, min(time_idx + self.num_frames, total_num_frames)):
                    img = f[str(idx_in_shard)][str(i)][()]
                    arr = torch.tensor(img)/255.0
                    prefinals.append(arr)

            data = torch.stack(prefinals)
            return data, video_index
    
def check_video_data_structure():
    import mediapy as media

    # dataset_root = "/mnt/sda/hjy/fdm/CARLA_Town_01_h5" # u11 - xs
    dataset_root = "/home/ubuntu/zzc/data/video_prediction/UCF101/UCF101_h5" # u11 - xs

    dataset_type = 'train'
    train_dataset = VideoDataset(dataset_root, dataset_type)
    print(len(train_dataset))
    print(train_dataset[10][0].shape)
    print(torch.min(train_dataset[10][0]), torch.max(train_dataset[10][0]))
    print(train_dataset[10][1])

    # dataset_type = 'valid'
    dataset_type = 'test'
    test_dataset = VideoDataset(dataset_root, dataset_type, total_videos=256)
    print(len(test_dataset))
    print(test_dataset[10][0].shape)
    print(torch.min(test_dataset[10][0]), torch.max(test_dataset[10][0]))
    print(test_dataset[10][1])
    
    train_video = train_dataset[20][0]
    test_video = test_dataset[20][0]
    
    train_video = dataset2video(train_video)
    test_video = dataset2video(test_video)
    
    print(train_video.shape)
    print(test_video.shape)

    media.show_video(rearrange(train_video, 't c h w -> t h w c').numpy(),fps = 20)
    media.show_video(rearrange(test_video, 't c h w -> t h w c').numpy(),fps = 20)

    """
    479
    torch.Size([40, 64, 64])
    tensor(0.0627) tensor(0.8078)
    10

    or like
    
    256
    torch.Size([30, 128, 128, 3])
    tensor(0.) tensor(0.8863)
    60

    """

def check_num_workers():
    from time import time
    import multiprocessing as mp
    from torch.utils.data import DataLoader

    print(f"num of CPU: {mp.cpu_count()}")

    # dataset_root = "/mnt/rhdd/zzc/data/video_prediction/KTH/processed/" # u8 - xs
    dataset_root = "/mnt/sda/hjy/kth/processed/" # u11 - xs
    # dataset_root = "/mnt/sda/hjy/kth/kth_h5/" # u16 - 0.72s
    dataset_type = 'train'
    train_dataset = VideoDataset(dataset_root, dataset_type)

    for num_workers in range(8, 10, 2):  
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

        for _ in range(5):
            start = time()
            for _, _ in enumerate(train_dataloader, 0):
                pass
            end = time()
            print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

if __name__ == "__main__":
    check_video_data_structure()
    # check_num_workers()
    

