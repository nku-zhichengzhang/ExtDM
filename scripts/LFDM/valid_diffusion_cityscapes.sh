# sh ./scripts/LFDM/valid_diffusion_cityscapes.sh

FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/flow_pretrained/old
seed=1234

while [ $seed -le 2000 ]
do
    CUDA_VISIBLE_DEVICES='1' \
    python ./scripts/LFDM/valid.py \
        --total_pred_frames 28 \
        --num_videos 256 \
        --valid_batch_size 64 \
        --ddim_sampling_steps 100 \
        --config ./config/LFDM/cityscapes128_origin.yaml \
        --flowae_checkpoint $FLOWCKPT/cityscapes128_perspective/snapshots/RegionMM.pth \
        --checkpoint ./logs_training/LFDM/cityscapes128_origin_LFDM_Batch64_lr2e-4_c2p5/snapshots/flowdiff_0064_S075000_207.109.pth \
        --log_dir ./logs_validation/LFDM/cityscapes128_origin_LFDM_Batch64_lr2e-4_c2p5 \
        --random_time False \
        --random-seed $seed
    seed=$((seed + 1000))
done