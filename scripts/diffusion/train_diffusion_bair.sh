# sh ./scripts/diffusion/train_diffusion_bair.sh

FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/flow_pretrained

# 从头训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint $FLOWCKPT/bair64/snapshots/RegionMM.pth \
    --config ./config/bair64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --random-seed 1234 \
    --postfix baseDM_wRefFea_wcond_woOutConv_Batch32_lr2e-4_c2p7

# 预训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint {$FLOWCKPT}/bair64/snapshots/RegionMM.pth \
#     --checkpoint ./logs_training/diffusion/kth64_test_without_rf/snapshots/flowdiff.pth \
#     --config ./config/bair64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix test_without_rf \
#     --set-start True
