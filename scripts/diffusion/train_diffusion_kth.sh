# sh ./scripts/diffusion/train_diffusion_kth.sh

# 从头训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/kth64/snapshots/RegionMM.pth \
    --config ./config/kth64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --postfix 0722

# 预训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/kth64/snapshots/RegionMM.pth \
#     --checkpoint ./logs_training/diffusion/kth64_0721_new/snapshots/flowdiff.pth \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix 0721_new \
#     --set-start True