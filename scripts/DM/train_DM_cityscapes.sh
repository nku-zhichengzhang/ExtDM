# sh ./scripts/DM/train_DM_cityscapes.sh

AE_CKPT_PATH=/home/ubuntu/zzc/data/video_prediction/AE_pretrained
AE_NAME=cityscapes128_perspective
AE_STEP=RegionMM
# AE_NAME=cityscapes128_FlowAE_Batch128_lr2e-4_Region40_perspective_scale0.50
# AE_STEP=RegionMM_0128_S100000
# AE_NAME=cityscapes128_FlowAE_Batch64_lr1e-4_Region40_perspective_scale1.00
# AE_STEP=RegionMM_best_123.506
SEED=1234

python ./scripts/DM/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT_PATH/Cityscapes/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/DM/cityscapes.yaml \
    --log_dir ./logs_training/DM/Cityscapes \
    --device_ids 0,1,2,3,4,5,6,7 \
    --postfix test