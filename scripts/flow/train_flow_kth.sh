# sh ./scripts/flow/train_flow_kth.sh

FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/pth # u8
# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/pth # u11
# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/pth # u16
# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/pth # hpc_403

# 从头训练
# python ./scripts/flow/run.py \
#     --checkpoint $FLOWCKPT/taichi256.pth \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --postfix FlowAE_Batch256_lr4e-4

# 从0训练
# python ./scripts/flow/run.py \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --random-seed 1234 \
#     --postfix FlowAE_Batch256_lr2e-4_Region20_affine

# 预训练
python ./scripts/flow/run.py \
    --checkpoint ./logs_training/flow/kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max50/snapshots/RegionMM_0256_S120000.pth \
    --config ./logs_training/flow/kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max50/kth64.yaml \
    --log_dir ./logs_training/flow \
    --device_ids 0,1 \
    --postfix FlowAE_Batch256_lr2e-4_Region20_affine_Max50 \
    --random-seed 1234 \
    --set-start True