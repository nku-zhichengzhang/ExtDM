# sh ./scripts/diffusion/train_diffusion_bair.sh

# 从头训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint ./logs_training/flow_pretrained/bair64/snapshots/RegionMM.pth \
    --config ./config/bair64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --random-seed 1234 \
    --postfix 0723_onlyflow

# 预训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint ./logs_training/flow_pretrained/kth64/snapshots/RegionMM.pth \
#     --checkpoint ./logs_training/diffusion/kth64_test_without_rf/snapshots/flowdiff.pth \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix test_without_rf \
#     --set-start True