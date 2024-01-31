# sh ./scripts/LFDM/train_diffusion_kth.sh

FLOWCKPT=/mnt/sda/hjy/data/flow_pretrained/old

python ./scripts/LFDM/run.py \
    --flowae_checkpoint $FLOWCKPT/kth64/snapshots/RegionMM.pth \
    --config ./config/LFDM/kth64_origin.yaml \
    --log_dir ./logs_training/LFDM \
    --device_ids 0,1 \
    --postfix LFDM_Batch32_lr2e-4_c10p5