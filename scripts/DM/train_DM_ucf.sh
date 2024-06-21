# sh ./scripts/DM/train_DM_ucf.sh

AE_CKPT_PATH=/home/ubuntu/zzc/data/video_prediction/AE_pretrained
AE_NAME=ucf101_64_FlowAE_Batch100_lr2e-4_Region64_scale0.5
AE_STEP=RegionMM_0100_S120000_270.85
# AE_NAME=ucf101_64_FlowAE_Batch100_lr2e-4_Region128_scale0.5
# AE_STEP=RegionMM_0100_S120000_236.946
SEED=1234

python ./scripts/DM/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT_PATH/UCF101/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/DM/ucf.yaml \
    --log_dir ./logs_training/DM/ucf \
    --device_ids 0,1 \
    --postfix test