# https://github.com/edenton/svg/blob/master/data/convert_bair.py
import argparse
import cv2
import glob
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm

sys.path.append("..") 
from h5 import HDF5Maker

class KTH_HDF5Maker(HDF5Maker):

    def add_video_info(self):
        pass

    def create_video_groups(self):
        self.writer.create_group('len')
        self.writer.create_group('videos')

    def add_video_data(self, data, dtype=None):
        frames = data
        self.writer['len'].create_dataset(str(self.count), data=len(frames))
        self.writer.create_group(str(self.count))
        for i, frame in enumerate(frames):
            self.writer[str(self.count)].create_dataset(str(i), data=frame, dtype=dtype, compression="lzf")

def read_data(video):
    # data torch [0,255] uint8 torch.Size([1000, 128, 128, 3])
    # return frames list  [0, 255] [ array, array, ...] [1000, 128, 128, 3]
    
    frames = []
    
    video = video.numpy()
    
    for frame in video:
        frames.append(frame)
    
    if frames == []:
        print("Error: ")
        print("1. clean old video file and old hdf5 file")
        print("2. download new video file")
        print("3. generate hdf5 file again")
        ValueError()
        
    return frames

# def show_video(frames):
#     import matplotlib.pyplot as plt
#     from matplotlib.animation import FuncAnimation
#     im1 = plt.imshow(frames[0])
#     def update(frame):
#         im1.set_data(frame)
#     ani = FuncAnimation(plt.gcf(), update, frames=frames, interval=10, repeat=False)
#     plt.show()

def make_h5_from_kth(carla_dir, out_dir='./h5_ds', vids_per_shard=1000000, force_h5=False):
    
    import pandas as pd
    import torch
    from tqdm import tqdm
    
    # train dataset    
    print(f"process train")
    dataste_dir = out_dir + '/train' 
    h5_maker = KTH_HDF5Maker(dataste_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)
    
    df = pd.read_csv(os.path.join(carla_dir, 'video_train.csv'))
    for i in tqdm(range(len(df))):
    # for i in tqdm(range(20)):
        video_filename = df['path'][i].split('/')[-1]
        video_filename = (os.path.join(carla_dir, video_filename))
        video = torch.load(video_filename)
        # [0-255] uint8 [1000, 128, 128, 3]
        frames = read_data(video)
        h5_maker.add_data(frames, dtype='uint8')
    h5_maker.close()
    
    # test dataset    
    print(f"process test")
    dataste_dir = out_dir + '/test' 
    h5_maker = KTH_HDF5Maker(dataste_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)
    
    df = pd.read_csv(os.path.join(carla_dir, 'video_test.csv'))
    for i in tqdm(range(len(df))):
        video_filename = df['path'][i].split('/')[-1]
        video_filename = (os.path.join(carla_dir, video_filename))
        video = torch.load(video_filename)
        # [0-255] uint8 [1000, 128, 128, 3]
        frames = read_data(video)
        h5_maker.add_data(frames, dtype='uint8')
    h5_maker.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--carla_dir', type=str, help="Directory with KTH")
    parser.add_argument('--vids_per_shard', type=int, default=1000000)
    parser.add_argument('--force_h5', type=eval, default=False)

    args = parser.parse_args()

    make_h5_from_kth(out_dir=args.out_dir, carla_dir=args.carla_dir, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)

# Example:
# cd /home/ubuntu/zzc/vidpred/CVPR23_LFDM/LFAE/data/CARLA_Town_01
# dataset_root=/mnt/hdd/zzc/data/video_prediction/CARLA_Town_01
# python 02_carla_town_01_convert.py --carla_dir $dataset_root/no-traffic --out_dir $dataset_root/CARLA_Town_01_h5 --force_h5 False
