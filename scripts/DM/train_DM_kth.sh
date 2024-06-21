# sh ./scripts/DM/train_DM_kth.sh

AE_CKPT_PATH=/home/ubuntu/zzc/data/video_prediction/AE_pretrained
AE_NAME=kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max40_2
AE_STEP=RegionMM_0256_S220000
# AE_NAME=kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max40_2
# AE_STEP=RegionMM_0256_S220000
# AE_NAME=kth64_FlowAE_Batch128_lr1e-4_Region20_affine_scale1.00_resume
# AE_STEP=RegionMM_best_157.143
SEED=1234

python ./scripts/DM/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT_PATH/KTH/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/DM/kth.yaml \
    --log_dir ./logs_training/DM/KTH \
    --device_ids 0,1,2,3 \
    --postfix test