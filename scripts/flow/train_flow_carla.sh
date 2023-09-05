# sh ./scripts/flow/train_flow_carla.sh

# 从头训练
# python ./scripts/flow/run.py \
#     --checkpoint /mnt/sda/hjy/pth/ted-youtube384.pth \
#     --config ./config/carla128.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --postfix test 

# 从0训练
python ./scripts/flow/run.py \
    --config ./config/carla128.yaml \
    --log_dir ./logs_training/flow \
    --device_ids 0,1 \
    --random-seed 1000 \
    --postfix FlowAE_Batch128_lr4e-5_Region20_perspective

# 预训练
# python ./scripts/flow/run.py \
#     --checkpoint ./logs_training/flow/carla128_test/snapshots/RegionMM.pth \
#     --config ./config/carla128.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --postfix test \
#     --set-start True