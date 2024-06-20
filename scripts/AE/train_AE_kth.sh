# sh ./scripts/AE/train_AE_kth.sh

# Training from scratch
python ./scripts/AE/run.py \
    --config ./config/AE/kth.yaml \
    --log_dir ./logs_training/AE/KTH \
    --device_ids 0,1 \
    --postfix test

# Resuming training from checkpoint
# --checkpoint ./logs_training/AE/<project_name>/snapshots/RegionMM.pth \
# --set-start True