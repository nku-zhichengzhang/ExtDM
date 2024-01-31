# sh ./scripts/flow/train_flow_ucf.sh

# 从头训练
# CUDA_VISIBLE_DEVICES=4 \
# python ./scripts/flow/run.py \
#     --config ./config/ucf101_64.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0 \
#     --random-seed 2024 \
#     --postfix FlowAE_Batch100_lr2e-4_Region32_scale0.5

# CUDA_VISIBLE_DEVICES=5 \
# python ./scripts/flow/run.py \
#     --checkpoint ./logs_training/flow/ucf101_64_FlowAE_Batch100_lr2e-4_Region32_perspective_scale0.5/snapshots/RegionMM_0100_S025000.pth \
#     --config ./config/ucf101_64.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0 \
#     --random-seed 428 \
#     --postfix FlowAE_Batch100_lr2e-4_Region32_perspective_scale0.5 \
#     --set-start True

# CUDA_VISIBLE_DEVICES=6 \
# python ./scripts/flow/run.py \
#     --config ./config/ucf101_64.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0 \
#     --random-seed 2024 \
#     --postfix FlowAE_Batch100_lr2e-4_Region64_scale0.5

CUDA_VISIBLE_DEVICES=2 \
python ./scripts/flow/run.py \
    --checkpoint ./logs_training/flow/ucf101_64_FlowAE_Batch100_lr2e-4_Region64_scale0.5/snapshots/RegionMM_0100_S120000_270.85.pth \
    --config ./config/ucf101_64.yaml \
    --log_dir ./logs_training/flow \
    --device_ids 0 \
    --random-seed 2048 \
    --postfix FlowAE_Batch100_lr2e-4_Region64_perspective_scale0.5