# sh ./scripts/diffusion/valid_diffusion_carla.sh

CUDA_VISIBLE_DEVICES=1 \
python ./scripts/diffusion/valid_carla.py \
    --total_cond_frames 36 \
    --total_pred_frames 964 \
    --num_videos 100 \
    --valid_batch_size 4 \
    --config ./config/carla128.yaml \
    --flowae_checkpoint /home/u1120230288/zzc/data/video_prediction/flow_pretrained/better/carla128_FlowAE_Batch128_lr1e-4_Region20_perspective_Max20_Seed1000_0903/snapshots/RegionMM_0128_S150000.pth \
    --checkpoint /home/u1120230288/zzc/code/video_prediction/EDM/logs_training/diffusion/carla128_DM_Batch32_lr2e-4_c10p5_STW_adaptor_multi_traj_ada/snapshots/flowdiff_best.pth \
    --log_dir ./logs_validation/diffusion/carla128_DM_Batch32_lr2e-4_c10p5_STW_adaptor_multi_traj_ada \
    --random_time False \
    --random-seed 1234 \
    --device_ids "0"

# CUDA_VISIBLE_DEVICES=0 \
# python ./scripts/diffusion/valid_carla.py \
#     --total_cond_frames 36 \
#     --total_pred_frames 964 \
#     --num_videos 100 \
#     --valid_batch_size 4 \
#     --config ./config/carla128.yaml \
#     --flowae_checkpoint /home/u1120230288/zzc/data/video_prediction/flow_pretrained/better/carla128_FlowAE_Batch128_lr1e-4_Region20_perspective_Max20_Seed1000_0903/snapshots/RegionMM_0128_S150000.pth \
#     --checkpoint /home/u1120230288/zzc/code/video_prediction/EDM/logs_training/diffusion/carla128_DM_Batch32_lr2e-4_c10p10_STW_adaptor_multi_traj_ada/snapshots/flowdiff_best_480.043.pth \
#     --log_dir ./logs_validation/diffusion/carla128_DM_Batch32_lr2e-4_c10p10_STW_adaptor_multi_traj_ada \
#     --random_time False \
#     --random-seed 1234 \
#     --device_ids "0"

# CUDA_VISIBLE_DEVICES=1 \
# python ./scripts/diffusion/valid_carla.py \
#     --total_cond_frames 36 \
#     --total_pred_frames 964 \
#     --num_videos 100 \
#     --valid_batch_size 4 \
#     --config ./config/carla128.yaml \
#     --flowae_checkpoint /home/u1120230288/zzc/data/video_prediction/flow_pretrained/better/carla128_FlowAE_Batch128_lr1e-4_Region20_perspective_Max20_Seed1000_0903/snapshots/RegionMM_0128_S150000.pth \
#     --checkpoint /home/u1120230288/zzc/code/video_prediction/EDM/logs_training/diffusion/carla128_DM_Batch32_lr2e-4_c10p20_STW_adaptor_scale0.25_multi_traj_ada/snapshots/flowdiff_best_614.218.pth \
#     --log_dir ./logs_validation/diffusion/carla128_DM_Batch32_lr2e-4_c10p20_STW_adaptor_scale0.25_multi_traj_ada \
#     --random_time False \
#     --random-seed 1234 \
#     --device_ids "0"