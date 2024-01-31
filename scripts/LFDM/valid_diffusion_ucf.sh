# sh ./scripts/LFDM/valid_diffusion_ucf.sh

DM_CKPT=/home/u1120230288/zzc/data/video_prediction/LFDM_pretrained
AE_CKPT=/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained

DATASET_PATH=/home/u1120230288/zzc/data/video_prediction/dataset
AE_NAME=ucf101_64_FlowAE_Batch100_lr2e-4_Region64_scale0.5
AE_STEP=RegionMM_0100_S120000_270.85
DM_NAME=ucf101_64_origin_LFDM_Batch16_lr2e-4_c4p8_scale0.50
DM_STEP=flowdiff_0016_S012000_796.425
SEED=1000
NUM_BATCH_SIZE=256

while [ $SEED -le 10000 ]
do
    CUDA_VISIBLE_DEVICES=2 \
    python ./scripts/LFDM/valid.py \
        --dataset_path $DATASET_PATH/UCF101_h5 \
        --pred_frames 4 \
        --total_pred_frames 16 \
        --num_videos 256 \
        --valid_batch_size $NUM_BATCH_SIZE \
        --ddim_sampling_steps 100 \
        --log_dir ./logs_validation/LFDM/UCF101/${DM_NAME}_${SEED} \
        --config $DM_CKPT/UCF101/$DM_NAME/ucf101_64_origin.yaml \
        --checkpoint $DM_CKPT/UCF101/$DM_NAME/snapshots/$DM_STEP.pth \
        --flowae_checkpoint $AE_CKPT/UCF101/$AE_NAME/snapshots/$AE_STEP.pth \
        --random-seed $SEED
    SEED=$((SEED + 1000))
done