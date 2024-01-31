# sh ./scripts/LFDM/valid_diffusion_kth.sh

DM_CKPT=/home/u1120230288/zzc/data/video_prediction/LFDM_pretrained
AE_CKPT=/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained

DATASET_PATH=/home/u1120230288/zzc/data/video_prediction/dataset
AE_NAME=kth64_region10_res0.5
AE_STEP=RegionMM
DM_NAME=videoflowdiff_kth
DM_STEP=flowdiff
SEED=1000
NUM_BATCH_SIZE=64

while [ $SEED -le 10000 ]
do
    CUDA_VISIBLE_DEVICES=4 \
    python ./scripts/LFDM/valid.py \
        --dataset_path $DATASET_PATH/KTH_h5 \
        --pred_frames 20 \
        --total_pred_frames 40 \
        --num_videos 256 \
        --valid_batch_size $NUM_BATCH_SIZE \
        --ddim_sampling_steps 100 \
        --log_dir ./logs_validation/LFDM/KTH/${DM_NAME}_${SEED} \
        --config $DM_CKPT/KTH/$DM_NAME/kth64_origin.yaml \
        --checkpoint $DM_CKPT/KTH/$DM_NAME/snapshots/$DM_STEP.pth \
        --flowae_checkpoint $AE_CKPT/KTH/$AE_NAME/snapshots/$AE_STEP.pth \
        --random-seed $SEED
    SEED=$((SEED + 1000))
done