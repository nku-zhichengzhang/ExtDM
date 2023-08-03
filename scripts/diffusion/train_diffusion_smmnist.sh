# sh ./scripts/diffusion/train_diffusion_smmnist.sh

FLOWCKPT=/mnt/sda/hjy/flow_pretrained

# 从头训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint $FLOWCKPT/smmnist64/snapshots/RegionMM.pth \
    --config ./config/smmnist64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --random-seed 1234 \
    --postfix DM_Batch32_lr2e-4_c10p5

# 从头训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint $FLOWCKPT/smmnist64/snapshots/RegionMM.pth \
#     --config ./config/smmnist64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --random-seed 1234 \
#     --postfix baseDM_Batch32_lr2e-4_c10p5

# 预训练
python ./scripts/diffusion/run.py \
    --set-start True \
    --flowae_checkpoint $FLOWCKPT/smmnist64/snapshots/RegionMM.pth \
    --checkpoint ./logs_training/diffusion/smmnist64_baseDM_Batch32_lr2e-4_c10p5/snapshots/flowdiff_0032_S045000.pth \
    --config ./logs_training/diffusion/smmnist64_baseDM_Batch32_lr2e-4_c10p5/smmnist64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --random-seed 1234 \
    --postfix baseDM_Batch32_lr2e-4_c10p5