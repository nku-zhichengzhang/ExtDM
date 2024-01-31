# sh ./scripts/output_setting/valid_diffusion_kth.sh

python ./scripts/output_setting/valid.py \
    --total_pred_frames 40 \
    --num_videos 256 \
    --valid_batch_size 128 \
    --flowae_checkpoint /home/u1120230288/zzc/data/video_prediction/flow_pretrained/better/kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max40_2/snapshots/RegionMM_0256_S220000.pth \
    --config ./output_model/kth64_DM_Batch32_lr2e-4_c10p5_STW_adaptor_multi_traj_ada/kth64.yaml \
    --checkpoint ./output_model/kth64_DM_Batch32_lr2e-4_c10p5_STW_adaptor_multi_traj_ada/flowdiff_best.pth \
    --log_dir ./logs_validation/diffusion/kth64_DM_Batch32_lr2e-4_c10p5_STW_adaptor_multi_traj_ada \
    --random_time True \
    --random-seed 1000 \
    --device_ids "0"