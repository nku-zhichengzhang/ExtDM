# sh ./scripts/diffusion/train_diffusion_carla.sh

# 从头训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint ./logs_training/flow_pretrained/carla128_20region/snapshots/RegionMM.pth \
#     --config ./config/carla128.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix not_onlyflow

# 预训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint ./logs_training/flow_pretrained/carla128_20region/snapshots/RegionMM.pth \
    --checkpoint ./logs_training/diffusion/carla128_not_onlyflow/snapshots/flowdiff.pth \
    --config ./config/carla128.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --postfix not_onlyflow \
    --set-start True