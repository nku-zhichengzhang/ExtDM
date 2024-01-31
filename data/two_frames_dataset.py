# build MHAD dataset for RegionMM

import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
from data.augmentation import AllAugmentationTransform
import cv2
from data.h5 import HDF5Dataset
from torchvision import transforms

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


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)]
        if frame_shape is not None:
            video_array = np.array([resize(frame, frame_shape) for frame in video_array])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if frame_shape is None:
            raise ValueError('Frame shape can not be None for stacked png format.')

        frame_shape = tuple(frame_shape)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape + (3, ))
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov') or name.lower().endswith('.avi'):
        video = mimread(name)
        if len(video[0].shape) == 2:
            video = [gray2rgb(frame) for frame in video]
        if frame_shape is not None:
            video = np.array([resize(frame, frame_shape) for frame in video])
        video = np.array(video)
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

class TwoFramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, 
                 root_dir, 
                 type='train', 
                 total_videos=-1, 
                 frame_shape=64, 
                 max_frame_distance=1000, 
                 min_frame_distance=10,
                 augmentation_params=None):
        self.root_dir = root_dir
        self.type = type
        self.frame_shape = frame_shape
        self.total_videos = total_videos
        self.max_frame_distance = max_frame_distance
        self.min_frame_distance = min_frame_distance

        if "UCF" in self.root_dir:
            self.videos_ds = HDF5Dataset(self.root_dir)
            # Train
            # self.num_train_vids = 9624
            # self.num_test_vids = 3696   # -> 369 : https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
            with self.videos_ds.opener(self.videos_ds.shard_paths[0]) as f:
                self.num_train_vids = f['num_train'][()]
                self.num_test_vids = f['num_test'][()]//10  # https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
        else:
            self.videos_ds = HDF5Dataset(os.path.join(self.root_dir, type))
        
        self.transform = AllAugmentationTransform(**augmentation_params)
        
    def __len__(self):
        if self.total_videos > 0:
            return self.total_videos
        else:
            if "UCF" in self.root_dir:
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
        if "UCF" in self.root_dir:
            return self.num_train_vids if self.type=='train' else self.num_test_vids
        else:  
            return len(self.videos_ds)
    
    # def get_frames(self, shard_idx, idx_in_shard, frame_idxs):
    #     frames = []
    #     with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
    #         for frame_idx in frame_idxs:
    #             frame = f[str(idx_in_shard)][str(frame_idx)][()]
    #             if len(frame.shape) == 2 or frame.shape[2] == 1:
    #                 frames.append(gray2rgb(frame))
    #             else:
    #                 frames.append(frame)
    #     return frames
    
    def __getitem__(self, index):
        if "UCF" in self.root_dir:
            video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))

            if not self.type=='train':
                video_index = video_index * 10 + self.num_train_vids    # https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
            shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

            crop_c = np.random.randint(int(self.frame_shape/240*320) - self.frame_shape) if self.type=='train' else int((self.frame_shape/240*320 - self.frame_shape)/2)

            with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
                num_frames = f['len'][str(idx_in_shard)][()]

                # 抽取两帧
                frame_idxs = np.sort(np.random.choice(num_frames, replace=True, size=2))
                while (frame_idxs[1] - frame_idxs[0] < self.min_frame_distance) or (frame_idxs[1] - frame_idxs[0] > self.max_frame_distance):
                    frame_idxs = np.sort(np.random.choice(num_frames, replace=True, size=2))
                
                video_array = []
                for frame_idx in frame_idxs:
                    frame = f[str(idx_in_shard)][str(frame_idx)][()]
                    frame = frame[:, crop_c:crop_c + self.frame_shape]
                    if len(frame.shape) == 2 or frame.shape[2] == 1:
                        video_array.append(gray2rgb(frame))
                    else:
                        video_array.append(frame)
        else:
            video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
            shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
            
            with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
                num_frames = f['len'][str(idx_in_shard)][()]

                # 抽取两帧
                frame_idxs = np.sort(np.random.choice(num_frames, replace=True, size=2))
                while (frame_idxs[1] - frame_idxs[0] < self.min_frame_distance) or (frame_idxs[1] - frame_idxs[0] > self.max_frame_distance):
                    frame_idxs = np.sort(np.random.choice(num_frames, replace=True, size=2))
                
                video_array = []
                for frame_idx in frame_idxs:
                    frame = f[str(idx_in_shard)][str(frame_idx)][()]
                    if len(frame.shape) == 2 or frame.shape[2] == 1:
                        video_array.append(gray2rgb(frame))
                    else:
                        video_array.append(frame)

        video_array = self.transform(video_array)

        out = {}

        source = np.array(video_array[0], dtype='float32')
        driving = np.array(video_array[1], dtype='float32')

        out['source'] = source.transpose((2, 0, 1))
        out['driving'] = driving.transpose((2, 0, 1))
        out['basename'] = shard_idx
        out['name'] = idx_in_shard
        out['frame'] = frame_idxs
        out['id'] = index
        
        return out

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]

if __name__ == "__main__":
    import yaml

    yaml_files = [
        # "/home/ubuntu11/zzc/code/videoprediction/EDM/config/smmnist64.yaml",
        # "/home/ubuntu11/zzc/code/videoprediction/EDM/config/kth64.yaml",
        # "/home/ubuntu11/zzc/code/videoprediction/EDM/config/bair64.yaml",
        # "/home/ubuntu11/zzc/code/videoprediction/EDM/config/cityscapes128.yaml",
        # "/home/ubuntu11/zzc/code/videoprediction/EDM/config/carla128.yaml",
        "/home/ubuntu/zzc/code/EDM_hpc/config/ucf101_64.yaml"
    ]

    total_videos = -1
    # total_videos = 256
    # total_videos = 100

    for yaml_file in yaml_files:

        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        dataset_params = config['dataset_params']

        dataset = TwoFramesDataset(
            root_dir=dataset_params['root_dir'], 
            type=dataset_params['train_params']['type'], 
            total_videos=total_videos, 
            frame_shape=dataset_params['frame_shape'], 
            max_frame_distance=dataset_params['max_frame_distance'], 
            min_frame_distance=dataset_params['min_frame_distance'],
            augmentation_params=dataset_params['augmentation_params']
        )
        
        print(len(dataset))
        print("*"*20)
        data = dataset[10]
        print("source ", data['source'].shape)
        print("driving", data['driving'].shape)
        print("frame  ", data['frame'])
        print("name   ", data['name'])
        print("id     ", data['id'])
        print("dataset len", len(dataset))

        import mediapy as media
        import einops

        media.show_images(
            [
                einops.rearrange(data['source'], "c h w -> h w c").squeeze(),
                einops.rearrange(data['driving'], "c h w -> h w c").squeeze(),
            ]
        )
