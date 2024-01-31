# sh ./scripts/diffusion/valid_diffusion_carla.sh

python ./scripts/diffusion/valid.py \
    --total_pred_frames 28 \
    --num_videos 256 \
    --valid_batch_size 64 \
    --config ./logs_training/flow/carla128_FlowAE_Batch128_lr1e-4_Region20_perspective_Max20_Seed1000_0903/carla128.yaml \
    --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/smmnist64/snapshots/RegionMM.pth \
    --checkpoint ./logs_training/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5/snapshots/flowdiff.pth \
    --log_dir ./logs_validation/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5 \
    --random_time False \
    --random-seed 1000 \
    --device_ids "0"