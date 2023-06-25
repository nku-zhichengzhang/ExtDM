# sh ./scripts/diffusion/train_diffusion_kth.sh

# 从头训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint ./logs_training/flow/kth64_test/snapshots/RegionMM.pth \
    --config ./config/kth64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --postfix test

# 预训练
# python ./scripts/flow/run.py \
#     --flowae_checkpoint ./logs_training/cityscapes128raw/snapshots/RegionMM.pth \
#     --checkpoint "" \
#     --config ./config/smmnist64.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --postfix test \
#     --set-start True