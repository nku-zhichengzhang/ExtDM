# sh ./scripts/flow/train_flow_cityscape.sh

# 从头训练
python ./scripts/flow/run.py \
    --checkpoint /mnt/sda/hjy/pth/ted-youtube384.pth \
    --config ./config/cityscapes128.yaml \
    --log_dir ./logs_training/flow \
    --device_ids 0,1 \
    --postfix test 

# 预训练
# python ./scripts/flow/run.py \
#     --checkpoint ./logs_training/flow/cityscapes128_test/snapshots/RegionMM.pth \
#     --config ./config/cityscapes128.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --postfix test \
#     --set-start True