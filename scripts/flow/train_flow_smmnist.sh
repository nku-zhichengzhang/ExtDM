# sh ./scripts/flow/train_flow_smmnist.sh

# 从头训练
python ./scripts/flow/run.py \
    --checkpoint /mnt/sda/hjy/pth/taichi256.pth \
    --config ./config/smmnist64.yaml \
    --log_dir ./logs_training/flow \
    --device_ids 0,1 \
    --postfix test 

# 预训练
# python ./scripts/flow/run.py \
#     --checkpoint ./logs_training/flow/smmnist64_test/snapshots/RegionMM.pth \
#     --config ./config/smmnist64.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --postfix test \
#     --set-start True