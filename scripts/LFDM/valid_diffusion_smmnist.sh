# sh ./scripts/LFDM/valid_diffusion_smmnist.sh

DM_CKPT=/home/u1120230288/zzc/data/video_prediction/LFDM_pretrained
AE_CKPT=/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained

DATASET_PATH=/home/u1120230288/zzc/data/video_prediction/dataset
AE_NAME=smmnist64_scale0.50
AE_STEP=RegionMM
DM_NAME=smmnist64_origin_LFDM_Batch32_lr2e-4_c10p5
DM_STEP=flowdiff_0032_S115000_21.321
SEED=1000
NUM_BATCH_SIZE=128

while [ $SEED -le 10000 ]
do
    CUDA_VISIBLE_DEVICES=3 \
    python ./scripts/LFDM/valid.py \
        --dataset_path $DATASET_PATH/SMMNIST_h5 \
        --pred_frames 5 \
        --total_pred_frames 10 \
        --num_videos 256 \
        --valid_batch_size $NUM_BATCH_SIZE \
        --ddim_sampling_steps 100 \
        --log_dir ./logs_validation/LFDM/SMMNIST/${DM_NAME}_${SEED} \
        --config $DM_CKPT/SMMNIST/$DM_NAME/smmnist64_origin.yaml \
        --checkpoint $DM_CKPT/SMMNIST/$DM_NAME/snapshots/$DM_STEP.pth \
        --flowae_checkpoint $AE_CKPT/SMMNIST/$AE_NAME/snapshots/$AE_STEP.pth \
        --random-seed $SEED
    SEED=$((SEED + 1000))
done