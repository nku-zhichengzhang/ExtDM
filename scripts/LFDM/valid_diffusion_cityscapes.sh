# sh ./scripts/LFDM/valid_diffusion_cityscapes.sh

DM_CKPT=/home/u1120230288/zzc/data/video_prediction/LFDM_pretrained
AE_CKPT=/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained

DATASET_PATH=/home/u1120230288/zzc/data/video_prediction/dataset
AE_NAME=cityscapes128_perspective
AE_STEP=RegionMM
DM_NAME=cityscapes128_origin_LFDM_Batch64_lr2e-4_c2p5
DM_STEP=flowdiff_0064_S075000_207.109
SEED=1000
NUM_BATCH_SIZE=256

while [ $SEED -le 10000 ]
do
    CUDA_VISIBLE_DEVICES=1 \
    python ./scripts/LFDM/valid.py \
        --dataset_path $DATASET_PATH/Cityscapes_h5 \
        --pred_frames 5 \
        --total_pred_frames 28 \
        --num_videos 256 \
        --valid_batch_size $NUM_BATCH_SIZE \
        --ddim_sampling_steps 100 \
        --log_dir ./logs_validation/LFDM/Cityscapes/${DM_NAME}_${SEED} \
        --config $DM_CKPT/Cityscapes/$DM_NAME/cityscapes128_origin.yaml \
        --checkpoint $DM_CKPT/Cityscapes/$DM_NAME/snapshots/$DM_STEP.pth \
        --flowae_checkpoint $AE_CKPT/Cityscapes/$AE_NAME/snapshots/$AE_STEP.pth \
        --random-seed $SEED
    SEED=$((SEED + 1000))
done