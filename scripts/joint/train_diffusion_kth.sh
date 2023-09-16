# sh ./scripts/diffusion/train_diffusion_kth.sh

# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/flow_pretrained # u8
FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u11
# FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u16
# FLOWCKPT=/home/u009079/zzc/data/vidp/flow_pretrained # hpc_403

# 从头训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint $FLOWCKPT/kth64/snapshots/RegionMM.pth \
    --config ./config/kth64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --postfix DM_Batch32_lr2e-4_c10p5

# 预训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/kth64/snapshots/RegionMM.pth \
#     --checkpoint ./logs_training/diffusion/kth64_0721_new/snapshots/flowdiff.pth \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix 0721_new \
#     --set-start True