# FlowDM: Diffusion Models in Compressed Motion Space for Video Prediction

## Install

```
conda env create -f environment.yml
conda activate edm
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