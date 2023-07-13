# sh ./scripts/diffusion/train_diffusion_cityscapes.sh

# 从头训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint ./logs_training/flow_pretrained/cityscapes128_perspective/snapshots/RegionMM.pth \
    --config ./config/cityscapes128.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --postfix not_onlyflow

# 预训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint ./logs_training/flow_pretrained/kth64/snapshots/RegionMM.pth \
#     --checkpoint ./logs_training/diffusion/kth64_test_without_rf/snapshots/flowdiff.pth \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix test_without_rf \
#     --set-start True