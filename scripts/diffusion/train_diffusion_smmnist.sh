# sh ./scripts/diffusion/train_diffusion_smmnist.sh

# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/flow_pretrained # u8
FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u11
# FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u16
# FLOWCKPT=/home/u009079/zzc/data/vidp/flow_pretrained # hpc_403

# 从头训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint $FLOWCKPT/smmnist64/snapshots/RegionMM.pth \
#     --config ./config/smmnist64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --random-seed 1234 \
#     --postfix DM_Batch32_lr2e-4_c10p4

# 预训练
python ./scripts/diffusion/run.py \
    --set-start True \
    --flowae_checkpoint $FLOWCKPT/smmnist64/snapshots/RegionMM.pth \
    --checkpoint ./logs_training/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p4/snapshots/flowdiff_0032_S130000.pth \
    --config ./logs_training/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p4/smmnist64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --random-seed 1234 \
    --postfix DM_Batch32_lr2e-4_c10p4