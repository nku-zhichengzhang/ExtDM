# sh ./scripts/flow/train_flow_cityscape.sh

CKPTPATH=/mnt/sda/hjy/pth

# 从头训练
python ./scripts/flow/run.py \
    --checkpoint $CKPTPATH/ted-youtube384.pth \
    --config ./config/cityscapes128.yaml \
    --log_dir ./logs_training/flow \
    --device_ids 0,1 \
    --postfix FlowAE_Batch128_lr2e-4_Region20_affine

# 从0训练
# python ./scripts/flow/run.py \
#     --config ./config/cityscapes128.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --postfix perspective

# 预训练
# python ./scripts/flow/run.py \
#     --checkpoint ./logs_training/flow/cityscapes128_test/snapshots/RegionMM.pth \
#     --config ./config/cityscapes128.yaml \
#     --log_dir ./logs_training/flow \
#     --device_ids 0,1 \
#     --postfix test \
#     --set-start True