<div align=center>
    <img src="assets/logo.png" width=15%>
    <h1>ExtDM: Distribution Extrapolation </br> Diffusion Model for Video Prediction</h1>

[**Zhicheng Zhang**](https://zzcheng.top/)<sup>1,2#</sup> · 
[**Junyao Hu**](https://junyaohu.github.io/)<sup>1,2#</sup> · 
[**Wentao Cheng**](https://wtchengcv.github.io/)<sup>1*</sup> · 
[**Danda Paudel**](https://people.ee.ethz.ch/~paudeld/)<sup>3,4</sup> · 
[**Jufeng Yang**](https://cv.nankai.edu.cn/)<sup>1,2</sup>

<sup>1</sup> VCIP & TMCC & DISSec, College of Computer Science, Nankai University

<sup>2</sup> Nankai International Advanced Research Institute (SHENZHEN · FUTIAN)

<sup>3</sup> Computer Vision Lab, ETH Zurich&ensp;&ensp;&ensp;&ensp;&ensp;<sup>4</sup> INSAIT, Sofia University

<sup>#</sup> Equal Contribution&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;<sup>+</sup> Corresponding Author

**🎉 Accepted by [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_ExtDM_Distribution_Extrapolation_Diffusion_Model_for_Video_Prediction_CVPR_2024_paper.html) 🎉**

[📃 [Paper](https://zzcheng.top/assets/pdf/2024_CVPR_ExtDM.pdf) ]
[📃 [中译版](https://zzcheng.top/assets/pdf/2024_CVPR_ExtDM_chinese.pdf) ]
[📦 [Code](https://github.com/nku-zhichengzhang/ExtDM) ]
[⚒️ [Project](https://zzcheng.top/ExtDM) ]
[📊 [Poster](https://zzcheng.top/assets/pdf/2024_CVPR_ExtDM_poster.pdf) ]
[📅 [Slide](https://zzcheng.top/assets/pdf/2024_CVPR_ExtDM_slide.pdf) ]
[🎞️ [Bilibili](https://www.bilibili.com/video/BV1dC411E72q) / [YouTube](https://www.youtube.com/watch?v=1hxOUagr8mM) ]

<img src="assets/demo.gif" width=400 />
</div>

> **TL;DR**: We present ExtDM, a new diffusion model that extrapolates video content from current frames by accurately modeling distribution shifts towards future frames.

## 📈 1. News

- 🔥2024-06-19: The code, datasets, and model weights are releasing!
- 2024-03-21: Creating repository. The code is coming soon ...
- 2024-02-27: ExtDM has been accepted to CVPR 2024！

## ⚒️ 2. Environment Setup

```
conda create -n ExtDM python=3.9
conda activate ExtDM
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install einops einops_exts rotary_embedding_torch==rotary-embedding-torch timm==0.4.5 imageio scikit-image opencv-python flow_vis matplotlib mediapy lpips h5py PyYAML tqdm wandb scipy==1.9.3
conda install ffmpeg
cd <your_path>/ExtDM
pip install -e .
```

Note: If you encounter the following error when running the code:

```
/home/<user_name>/anaconda3/envs/ExtDM/bin/ffmpeg: error while loading shared libraries: libopenh264.so.5: cannot open shared object file: No such file or directory
```

you can solve it by make a copy file like:

```
cp /home/<user_name>/anaconda3/envs/ExtDM/lib/libopenh264.so.6 /home/<user_name>/anaconda3/envs/ExtDM/lib/libopenh264.so.5
```

## 💾 3. Datasets

**Overview of the preprocessed dataset:**

| Dataset    | Len (train) | Len (test)           | Avg. Frames (train) | Setup (c->p) | Link & Size|
| ---------- | ----------- | -------------------- | ---------------------- | ---------------- | ------------------------------------------------------------ |
| SMMNIST    | 60000       | 256                  | 40                     | 10 -> 10         | [google drive](https://drive.google.com/file/d/1nn3yxrKLcRwAkmbHjNQOXuJBTFJzKpIe/view?usp=sharing) (688M) |
| KTH        | 479         | 120 (sample to 256)  | 483.18                 | 10 -> 30/40      | [google drive](https://drive.google.com/file/d/1h7eOcq-j1hJSrIX6s60T5E0tydI3vbQX/view?usp=sharing) (919M) |
| BAIR       | 43264       | 256                  | 30                     | 2 -> 14/28       | [google drive](https://drive.google.com/file/d/1-R_srAOy5ZcylGXVernqE4WLCe6N4_wq/view?usp=sharing) (13G) |
| Cityscapes | 2975        | 1525 (sample to 256) | 30                     | 2 -> 28          | [google drive](https://drive.google.com/file/d/1oP7n-FUfa9ifsMn6JHNS9depZfftvrXx/view?usp=sharing) (1.3G) |
| UCF-101    | -           | -                    | -                      | 4 -> 12          | [google drive](https://drive.google.com/file/d/1bDqhhfKYrdbIIOZeJcWHWjSyFQwmO1t-/view?usp=sharing) (40G) |


### 📅 3.1 Stochastic Moving MNIST (SMMNIST, 64x64, ch1)

This script will automatically download the PyTorch MNIST dataset, which will be used to dynamically generate random move MNIST. The script will save the randomly generated content in the HDF5 dataset format.

**How the data was processed:**

```
cd <your_path>/ExtDM/data/SMMNIST
dataset_root=<your_data_path>/SMMNIST
python 01_mnist_download_and_convert.py --image_size 64 --mnist_dir $dataset_root --out_dir $dataset_root/processed --force_h5 False
```

### 📅 3.2 KTH (64x64, ch1)

**How the data was processed:**

```
cd <your_path>/ExtDM/data/KTH
dataset_root=<your_data_path>/KTH
sh 01_kth_download.sh $dataset_root
python 02_kth_train_val_test_split.py
python 03_kth_convert.py --split_dir ./ --image_size 64 --kth_dir $dataset_root/raw --out_dir $dataset_root/mixed_processed --force_h5 False
```

### 📅 3.3 BAIR (64x64, ch3)

**How the data was processed:**

```
cd <your_path>/ExtDM/data/BAIR
dataset_root=<your_data_path>/BAIR
sh 01_bair_download.sh $dataset_root
python bair_convert.py --bair_dir $dataset_root/raw --out_dir $dataset_root/processed
```

### 📅 3.4 Cityscapes (64x64, ch3)

**How the data was processed:**

MAKE SURE YOU HAVE ~657GB SPACE! 324GB for the zip file, and 333GB for the unzipped image files

1. Download Cityscapes video dataset (`leftImg8bit_sequence_trainvaltest.zip` (324GB)) :\
`sh cityscapes_download.sh username password`\
using your `username` and `password` that you created on https://www.cityscapes-dataset.com/
2. Convert it to HDF5 format, and save in `/path/to/Cityscapes<image_size>_h5`:\
`python datasets/cityscapes_convert.py --leftImg8bit_sequence_dir '/path/to/Cityscapes/leftImg8bit_sequence' --image_size 64 --out_dir '/path/to/Cityscapes64_h5'`

### 📅 3.5 UCF-101 (orig:320x240, ch3)

**How the data was processed:**

MAKE SURE YOU HAVE ~20GB SPACE! 6.5GB for the zip file, and 8GB for the unzipped image files
1. Download UCF-101 video dataset (`UCF101.rar` (6.5GB)) :\
`sh cityscapes_download.sh /download/dir`
2. Convert it to HDF5 format, and save in `/path/to/UCF101_h5`:\
`python datasets/ucf101_convert.py --out_dir /path/to/UCF101_h5 --ucf_dir /download/dir/UCF-101 --splits_dir /download/dir/ucfTrainTestlist`

## 🧊 4. Checkpoints

### 🪄 4.1 AE Checkpoints

```
TODO
```

### 🪄 4.2 DM Checkpoints

```
TODO
```

## 🔬 5. Training & Inference

### 🔮 5.1: AE Training & Inference

**AE Training**

1. check `./config/AE/[DATASET].yaml`: set proper params for `root_dir`, `num_regions`, `max_epochs`, `num_repeats`, `lr`, `batch_size`, etc.

2. run `sh ./scripts/AE/train_AE_[DATASET].sh`

    ```
    sh ./scripts/AE/train_AE_smmnist.sh
    sh ./scripts/AE/train_AE_kth.sh
    sh ./scripts/AE/train_AE_bair.sh
    sh ./scripts/AE/train_AE_cityscapes.sh
    sh ./scripts/AE/train_AE_ucf.sh
    ```

3. you can see your running exp dir in `./logs_training/AE/[DATASET]/[EXP_NAME]`, or see details in wandb panels.

**AE Inference**

1. run `sh ./scripts/AE/train_AE_[DATASET].sh`

    ```
    sh ./scripts/AE/valid_AE_smmnist.sh
    sh ./scripts/AE/valid_AE_kth.sh
    sh ./scripts/AE/valid_AE_bair.sh
    sh ./scripts/AE/valid_AE_cityscapes.sh
    sh ./scripts/AE/valid_AE_ucf.sh
    ```

2. you can see your running exp dir in `./logs_validation/AE/[DATASET]/[EXP_NAME]`.

### 🔮 5.2: DM Training & Inference

**DM Training**

1. check `./config/DM/[DATASET].yaml`: set proper params for `root_dir`,  `max_epochs`, `num_repeats`, `lr`, `batch_size`, etc.

2. run `sh ./scripts/DM/train_DM_[DATASET].sh`

    ```
    sh ./scripts/DM/train_DM_smmnist.sh
    sh ./scripts/DM/train_DM_kth.sh
    sh ./scripts/DM/train_DM_bair.sh
    sh ./scripts/DM/train_DM_cityscapes.sh
    sh ./scripts/DM/train_DM_ucf.sh
    ```

3. you can see your running exp dir in `./logs_training/DM/[DATASET]/[EXP_NAME]`, or see details in wandb panels.

**DM Inference**

1. run `sh ./scripts/DM/train_DM_[DATASET].sh`

    ```
    sh ./scripts/DM/valid_DM_smmnist.sh
    sh ./scripts/DM/valid_DM_kth.sh
    sh ./scripts/DM/valid_DM_bair.sh
    sh ./scripts/DM/valid_DM_cityscapes.sh
    sh ./scripts/DM/valid_DM_ucf.sh
    ```

2. you can see your running exp dir in `./logs_validation/DM/[DATASET]/[EXP_NAME]`.

## ⭐ 6. Star History

[![Star History Chart](https://api.star-history.com/svg?repos=nku-zhichengzhang/ExtDM&type=Date)](https://star-history.com/#nku-zhichengzhang/ExtDM&Date)

## 📫 7. Contact

If you have any questions, please feel free to contact:

- Zhicheng Zhang: gloryzzc6@sina.com
- Junyao Hu: hujunyao@mail.nankai.edu.cn

## 🏷️ 8. Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{zhang2024ExtDM,
  title={ExtDM: Distribution Extrapolation Diffusion Model for Video Prediction},
  author={Zhang, Zhicheng and Hu, Junyao and Cheng, Wentao and Paudel, Danda and Yang, Jufeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (CVPR)},
  year={2024}
}
```

## 🥰 9. Acknowledgements

This code borrows from [CVPR23_LFDM](https://github.com/nihaomiao/CVPR23_LFDM) (by [@nihaomiao](https://github.com/nihaomiao)). The datasets partly comes from [mcvd-pytorch](https://github.com/voletiv/mcvd-pytorch) (by [@voletiv](https://github.com/voletiv)).
