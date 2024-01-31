# sh ./scripts/LFDM/train_diffusion_cityscapes.sh

# 从头训练
python ./scripts/LFDM/run.py \
    --flowae_checkpoint /mnt/rhdd/zzc/data/video_prediction/flow_pretrained/old/cityscapes128_perspective/snapshots/RegionMM.pth \
    --config ./config/LFDM/cityscapes128_origin.yaml \
    --log_dir ./logs_training/LFDM \
    --device_ids 0,1 \
    --postfix LFDM_Batch64_lr2e-4_c2p5