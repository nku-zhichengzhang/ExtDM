# sh ./scripts/LFDM/valid_diffusion_kth.sh

FLOWCKPT=/mnt/sda/hjy/data/flow_pretrained/old
seed=1000

while [ $seed -le 1000 ]
do
    CUDA_VISIBLE_DEVICES='1' \
    python ./scripts/LFDM/valid.py \
        --total_pred_frames 20 \
        --num_videos 16 \
        --valid_batch_size 4 \
        --ddim_sampling_steps 10 \
        --config ./config/LFDM/kth64_origin.yaml \
        --flowae_checkpoint $FLOWCKPT/kth64/snapshots/RegionMM.pth \
        --checkpoint ./logs_training/LFDM/kth64_origin_LFDM_Batch32_lr2e-4_c10p5/snapshots/flowdiff.pth \
        --log_dir ./logs_validation/LFDM/kth64_origin_LFDM_Batch32_lr2e-4_c10p5 \
        --random_time True \
        --random-seed $seed
    seed=$((seed + 1000))
done