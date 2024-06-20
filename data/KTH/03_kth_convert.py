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

# def show_video(frames):
#     import matplotlib.pyplot as plt
#     from matplotlib.animation import FuncAnimation
#     im1 = plt.imshow(frames[0])
#     def update(frame):
#         im1.set_data(frame)
#     ani = FuncAnimation(plt.gcf(), update, frames=frames, interval=10, repeat=False)
#     plt.show()

def make_h5_from_kth(kth_dir, split_dir, image_size=64, out_dir='./h5_ds', vids_per_shard=1000000, force_h5=False):
    
    # classes = ['train', 'valid', 'test']
    classes = ['train', 'valid']

    for type in classes:
        print(f"process {type}")
        dataste_dir = out_dir + '/' + type
        h5_maker = KTH_HDF5Maker(dataste_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)
        count = 0
        # try:
        with open(f"{split_dir}/{type}.txt", "r") as f:
            lines = f.read().splitlines()

        isSplit = len(lines[0].split(' ')) > 1
        
        for line in tqdm(lines):
            if isSplit:
                path, start, end = line.split(' ')
                path = os.path.join(kth_dir, path)
                frames = read_video(path, image_size)
                frames = frames[int(start):int(end)]
            else:
                path = os.path.join(kth_dir, line)
                frames = read_video(path, image_size)
            h5_maker.add_data(frames, dtype='uint8')
            count += 1
        # except StopIteration:
        #     break
        # except (KeyboardInterrupt, SystemExit):
        #     print("Ctrl+C!!")
        #     break
        # except:
        #     e = sys.exc_info()[0]
        #     print("ERROR:", e)

        h5_maker.close()
    print(f"process {type} done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--split_dir', type=str, help="Directory to split dataset files")
    parser.add_argument('--kth_dir', type=str, help="Directory with KTH")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--vids_per_shard', type=int, default=1000000)
    parser.add_argument('--force_h5', type=eval, default=False)

    args = parser.parse_args()

    make_h5_from_kth(out_dir=args.out_dir, kth_dir=args.kth_dir, split_dir=args.split_dir, image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)

# Example:

