# sh ./scripts/DM/train_DM_smmnist.sh

AE_CKPT_PATH=/home/ubuntu/zzc/data/video_prediction/AE_pretrained
AE_NAME=smmnist64_FlowAE_Batch100_lr2e-4_Region10_affine_scale0.50
AE_STEP=RegionMM
# AE_NAME=smmnist64_FlowAE_Batch128_lr1e-4_Region10_affine_scale1.00
# AE_STEP=RegionMM_best_2.183
SEED=1234

python ./scripts/DM/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT_PATH/SMMNIST/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/DM/smmnist.yaml \
    --log_dir ./logs_training/DM/SMMNIST \
    --device_ids 0,1,2,3 \
    --postfix test