import bisect
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import mediapy as media

import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# for image dataset
# import albumentations
# from PIL import Image
import torch

from data.h5 import HDF5Dataset

class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class VideoPaths(Dataset):
    def __init__(self, paths, start_idxs, end_idxs, trans=None, labels=None):
        self._length = len(paths)
        self._trans = trans

        if labels is None:
            self.labels = dict() 
        else:
            self.labels = labels

        self.labels["file_path"] = paths
        self.labels["start_idx"] = start_idxs
        self.labels["end_idx"] = end_idxs

    def __len__(self):
        return self._length

    def preprocess_video(self, video_path, start_idx, end_idx):
        video = media.read_video(video_path)[start_idx:end_idx]
        video = np.array(video).astype(np.uint8)
        tmp_video = []
        for i in range(len(video)):
            tmp_video.append(self._trans(image=video[i])["image"])
        video = np.array(tmp_video)
        video = (video/127.5 - 1.0).astype(np.float32)
        # [0,255] -> [-1,1]
        return video

    def __getitem__(self, i):
        video = dict()
        video["video"] = self.preprocess_video(self.labels["file_path"][i], int(self.labels["start_idx"][i]), int(self.labels["end_idx"][i]))
        for k in self.labels:
            video[k] = self.labels[k][i]
        return video
    

class HDF5InterfaceDataset(Dataset):
    def __init__(self, data_dir, frames_per_sample, random_time=True, total_videos=-1, start_at=0, labels=None):
        super().__init__()
        if labels is None:
            self.labels = dict() 
        else:
            self.labels = labels
        self.data_dir = data_dir
        self.videos_ds = HDF5Dataset(data_dir)
        self.total_videos = total_videos
        self.start_at = start_at
        self.random_time = random_time
        self.frames_per_sample = frames_per_sample

        # The numpy HWC image is converted to pytorch CHW tensor. 
        # If the image is in HW format (grayscale image), 、
        # it will be converted to pytorch HW tensor.
        flag = random.choice([0,1])

        self.trans = A.Compose([
            A.HorizontalFlip(p=flag),
            ToTensorV2()
        ])

    def __len__(self):
        if self.total_videos > 0:
            return self.total_videos
        else:
            return len(self.videos_ds)
        
    def max_index(self):
        return len(self.videos_ds)

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len
    
    def __getitem__(self, index, time_idx=0):
        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video = dict()

        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        final_clip = []
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()] - self.start_at
            if self.random_time and video_len > self.frames_per_sample:
                time_idx = np.random.choice(video_len - self.frames_per_sample)
            time_idx += self.start_at
            for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
                final_clip.append(self.trans(image=f[str(idx_in_shard)][str(i)][()])["image"])
        final_clip = torch.stack(final_clip)
        final_clip = (final_clip/127.5 - 1.0).type(torch.float32)
        video["video"] = final_clip

        for k in self.labels:
            video[k] = self.labels[k][i]

        return video


# class ImagePaths(Dataset):
#     def __init__(self, paths, size=None, random_crop=False, labels=None):
#         self.size = size
#         self.random_crop = random_crop

#         self.labels = dict() if labels is None else labels
#         self.labels["file_path"] = paths
#         self._length = len(paths)

#         if self.size is not None and self.size > 0:
#             self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
#             if not self.random_crop:
#                 self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
#             else:
#                 self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
#             self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
#         else:
#             self.preprocessor = lambda **kwargs: kwargs

#     def __len__(self):
#         return self._length

#     def preprocess_image(self, image_path):
#         image = Image.open(image_path)
#         if not image.mode == "RGB":
#             image = image.convert("RGB")
#         image = np.array(image).astype(np.uint8)
#         image = self.preprocessor(image=image)["image"]
#         image = (image/127.5 - 1.0).astype(np.float32)
#         return image

#     def __getitem__(self, i):
#         example = dict()
#         example["image"] = self.preprocess_image(self.labels["file_path_"][i])
#         for k in self.labels:
#             example[k] = self.labels[k][i]
#         return example
    
# class NumpyPaths(ImagePaths):
#     def preprocess_image(self, image_path):
#         image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
#         image = np.transpose(image, (1,2,0))
#         image = Image.fromarray(image, mode="RGB")
#         image = np.array(image).astype(np.uint8)
#         image = self.preprocessor(image=image)["image"]
#         image = (image/127.5 - 1.0).astype(np.float32)
#         return image
