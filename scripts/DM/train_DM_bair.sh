# sh ./scripts/DM/train_DM_bair.sh

AE_CKPT_PATH=/home/ubuntu/zzc/data/video_prediction/AE_pretrained
AE_NAME=bair64_scale0.50
AE_STEP=RegionMM
SEED=1234

python ./scripts/DM/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT_PATH/BAIR/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/DM/bair.yaml \
    --log_dir ./logs_training/DM/BAIR \
    --device_ids 0,1,2,3 \
    --postfix test