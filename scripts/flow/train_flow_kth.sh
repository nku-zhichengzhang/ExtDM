# sh ./scripts/flow/train_flow_kth.sh

FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/pth # u8
# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/pth # u11
# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/pth # u16
# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/pth # hpc_403

# 从头训练
python ./scripts/flow/run.py \
    --checkpoint $FLOWCKPT/taichi256.pth \
    --config ./config/kth64.yaml \
    --log_dir ./logs_training/flow \
    --device_ids 0,1 \
    --postfix FlowAE_Batch256_lr4e-4

# 预训练
# python ./scripts/flow/run.py \
#     --checkpoint ./logs_training/cityscapes128raw/snapshots/RegionMM.pth \
#     --config ./config/smmnist64.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --postfix test \
#     --set-start True