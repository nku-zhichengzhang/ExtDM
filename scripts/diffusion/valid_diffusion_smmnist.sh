# sh ./scripts/diffusion/valid_diffusion_smmnist.sh

python ./scripts/diffusion/valid.py \
    --total_pred_frames 10 \
    --num_videos 10 \
    --valid_batch_size 1 \
    --config ./logs_training/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5/smmnist64.yaml \
    --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/smmnist64/snapshots/RegionMM.pth \
    --checkpoint ./logs_training/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5/snapshots/flowdiff.pth \
    --log_dir ./logs_validation/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5 \
    --random_time False \
    --random-seed 1000 \
    --device_ids "0"