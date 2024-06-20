# sh ./scripts/AE/train_AE_ucf.sh

# Training from scratch
python ./scripts/AE/run.py \
    --config ./config/AE/ucf.yaml \
    --log_dir ./logs_training/AE/UCF \
    --device_ids 0,1 \
    --postfix test 

# Resuming training from checkpoint
# --checkpoint ./logs_training/AE/<project_name>/snapshots/RegionMM.pth \
# --set-start True