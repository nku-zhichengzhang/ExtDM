# FlowDM: Diffusion Models in Compressed Motion Space for Video Prediction

## Install

```
conda env create -f environment.yml
conda activate edm
pip install -e .
```

## Training 

### Step 1: FlowAE Training

- check `./config/[DATASET].yaml`
- run `sh ./scripts/flow/train_flow_[DATASET].sh`

### Step 2: FlowDM Training

- check `./config/[DATASET].yaml`
- run `sh ./scripts/diffusion/train_diffusion_[DATASET].sh`