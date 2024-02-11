# sh ./scripts/flow/train_flow_kth.sh

# 从头训练
# CUDA_VISIBLE_DEVICES=0 \
# python ./scripts/flow/run.py \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/flow/KTH/ \
#     --device_ids 0 \
#     --postfix FlowAE_Batch256_lr1e-4_region32_affine_scale0.5 \
#     --checkpoint ./logs_training/flow/KTH/kth64_FlowAE_Batch256_lr1e-4_region32_affine_scale0.5/snapshots/RegionMM_best_217.124.pth \
#     --set-start True

# CUDA_VISIBLE_DEVICES=1 \
# python ./scripts/flow/run.py \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/flow/KTH/ \
#     --device_ids 0 \
#     --postfix FlowAE_Batch256_lr1e-4_region64_affine_scale0.5 \
#     --checkpoint ./logs_training/flow/KTH/kth64_FlowAE_Batch256_lr1e-4_region64_affine_scale0.5/snapshots/RegionMM_best_229.734.pth \
#     --set-start True

# CUDA_VISIBLE_DEVICES=2 \
# python ./scripts/flow/run.py \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/flow/KTH/ \
#     --device_ids 0 \
#     --postfix FlowAE_Batch256_lr1e-4_region72_affine_scale0.5

CUDA_VISIBLE_DEVICES=3 \
python ./scripts/flow/run.py \
    --config ./config/kth64.yaml \
    --log_dir ./logs_training/flow/KTH/ \
    --device_ids 0 \
    --postfix FlowAE_Batch256_lr8e-5_region96_affine_scale0.5 \
    --checkpoint ./logs_training/flow/KTH/kth64_FlowAE_Batch256_lr2e-5_region96_affine_scale0.5/snapshots/RegionMM_best_17500_251.620.pth \
    --set-start True