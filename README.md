# FlowDM: Diffusion Models in Compressed Motion Space for Video Prediction

## Install

```
conda create -n EDM python=3.9
conda activate EDM
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install xformers==0.0.16 triton
pip install einops einops_exts rotary_embedding_torch timm==0.4.5
pip install imageio scikit-image opencv-python flow_vis matplotlib mediapy 
pip install h5py PyYAML tqdm wandb
pip install -e .
```

If you have a wrong with 

```
/home/xxx/anaconda3/envs/edm/bin/ffmpeg: error while loading shared libraries: libopenh264.so.5: cannot open shared object file: No such file or directory
```

you can solve it by make a copy file like 

```
cp /home/xxx/anaconda3/envs/edm/lib/libopenh264.so.6 /home/xxx/anaconda3/envs/edm/lib/libopenh264.so.5
```

## Training 

### Step 1: FlowAE Training

- check `./config/[DATASET].yaml`
- run `sh ./scripts/flow/train_flow_[DATASET].sh`

### Step 2: FlowDM Training

- check `./config/[DATASET].yaml`
- run `sh ./scripts/diffusion/train_diffusion_[DATASET].sh`
