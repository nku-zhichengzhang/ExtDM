# https://github.com/edenton/svg/blob/master/data/convert_bair.py
import argparse
import cv2
import glob
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm
from stochastic_moving_mnist import StochasticMovingMNIST
import torch

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

def read_video(video_path, image_size):
    # opencv is faster than mediapy
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(gray, (image_size, image_size))
        frames.append(image)
    cap.release()
    if frames == []:
        print(f"the file {video_path} may has been damaged, you should")
        print("1. clean old video file and old hdf5 file")
        print("2. download new video file")
        print("3. generate hdf5 file again")
        ValueError()
    return frames

def read_data(video):
    # data torch [0,1] torch.Size([40, 1, 64, 64])
    # return frames list  [0, 255] [ array, array, ...]
    
    frames = []
    
    video = (video.squeeze()*255).type(torch.uint8).numpy()
    
    for frame in video:
        frames.append(frame)
    
    if frames == []:
        print("Error: ")
        print("1. clean old video file and old hdf5 file")
        print("2. download new video file")
        print("3. generate hdf5 file again")
        ValueError()
        
    return frames

def make_h5_from_kth(mnist_dir, image_size=64, seq_len=40, out_dir='./h5_ds', vids_per_shard=1000000, force_h5=False):
    
    train_dataset = StochasticMovingMNIST(
        mnist_dir, train=True, seq_len=seq_len, num_digits=2,
        step_length=0.1, with_target=False
    )

    test_dataset = StochasticMovingMNIST(
        mnist_dir, train=False, seq_len=seq_len, num_digits=2,
        step_length=0.1, with_target=False, 
        total_videos=256
    )
    
    print(len(train_dataset))

    print(train_dataset[0].shape)
    print(train_dataset[0].shape)
    
    # import torch
    # print(torch.min(train_dataset[0][1]), torch.max(train_dataset[0][1]))
    # # value [0, 1]
    
    # train_dataset
    
    print(f"process train_dataset")
    
    dataste_dir = out_dir + '/train'
    h5_maker = KTH_HDF5Maker(dataste_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)
    
    
    for data in tqdm(train_dataset):
        # try:
        frames = read_data(data)
        h5_maker.add_data(frames, dtype='uint8')
        # except StopIteration:
        #     break
        # except (KeyboardInterrupt, SystemExit):
        #     print("Ctrl+C!!")
        #     break
        # except:
        #     e = sys.exc_info()[0]
        #     print("ERROR:", e)

    h5_maker.close()
    
    # test_dataset 
    
    print(f"process test_dataset")
    
    dataste_dir = out_dir + '/test'
    h5_maker = KTH_HDF5Maker(dataste_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)
    
    for data in tqdm(test_dataset):
        # try:
        frames = read_data(data)
        h5_maker.add_data(frames, dtype='uint8')
        # except StopIteration:
        #     break
        # except (KeyboardInterrupt, SystemExit):
        #     print("Ctrl+C!!")
        #     break
        # except:
        #     e = sys.exc_info()[0]
        #     print("ERROR:", e)

    h5_maker.close()
      
    print(f"process done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=40)
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--mnist_dir', type=str, help="Directory with KTH")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--vids_per_shard', type=int, default=1000000)
    parser.add_argument('--force_h5', type=eval, default=False)

    args = parser.parse_args()

    make_h5_from_kth(out_dir=args.out_dir, mnist_dir=args.mnist_dir, image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)