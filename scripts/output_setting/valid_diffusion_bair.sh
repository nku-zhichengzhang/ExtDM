# sh ./scripts/output_setting/valid_diffusion_bair.sh

python ./scripts/output_setting/valid.py \
    --total_pred_frames 28 \
    --num_videos 256 \
    --valid_batch_size 128 \
    --flowae_checkpoint /home/u1120230288/zzc/data/video_prediction/flow_pretrained/better/bair64/snapshots/RegionMM.pth \
    --config ./output_model/bair64_DM_Batch64_lr2.e-4_c2p5_STW_adaptor_multi_traj_resume/bair64.yaml \
    --checkpoint ./output_model/bair64_DM_Batch64_lr2.e-4_c2p5_STW_adaptor_multi_traj_resume/flowdiff_best.pth \
    --log_dir ./logs_validation/diffusion/bair64_DM_Batch64_lr2.e-4_c2p5_STW_adaptor_multi_traj_resume \
    --random_time False \
    --random-seed 1000 \
    --device_ids "0"