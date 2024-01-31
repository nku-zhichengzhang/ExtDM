# sh ./scripts/output_setting/valid_diffusion_cityscapes.sh

python ./scripts/output_setting/valid.py \
    --total_pred_frames 28 \
    --num_videos 256 \
    --valid_batch_size 64 \
    --flowae_checkpoint /home/u1120230288/zzc/data/video_prediction/flow_pretrained/better/cityscapes128_FlowAE_Batch128_lr2e-4_Region40_perspective_scale0.50/RegionMM_0128_S100000.pth \
    --config ./output_model/cityscapes128_DM_Batch28_lr1.5e-4_c2p7_STW_adaptor_scale0.5_multi_traj_ada/cityscapes128.yaml \
    --checkpoint ./output_model/cityscapes128_DM_Batch28_lr1.5e-4_c2p7_STW_adaptor_scale0.5_multi_traj_ada/flowdiff_best.pth \
    --log_dir ./logs_validation/diffusion/cityscapes128_DM_Batch28_lr1.5e-4_c2p7_STW_adaptor_scale0.5_multi_traj_ada \
    --random_time False \
    --random-seed 1000 \
    --device_ids "0"