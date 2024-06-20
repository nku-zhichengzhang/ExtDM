# sh ./scripts/AE/train_AE_smmnist.sh

# Training from scratch
python ./scripts/AE/run.py \
    --config ./config/AE/smmnist.yaml \
    --log_dir ./logs_training/AE/SMMNIST \
    --device_ids 0,1 \
    --postfix test 

# Resuming training from checkpoint
# --checkpoint ./logs_training/AE/<project_name>/snapshots/RegionMM.pth \
# --set-start True