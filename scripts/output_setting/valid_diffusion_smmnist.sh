# sh ./scripts/output_setting/valid_diffusion_smmnist.sh

python ./scripts/output_setting/valid.py \
    --total_pred_frames 10 \
    --num_videos 256 \
    --valid_batch_size 64 \
    --flowae_checkpoint /home/u1120230288/zzc/data/video_prediction/flow_pretrained/better/smmnist64/snapshots/RegionMM.pth \
    --config ./output_model/smmnist64_DM_Batch32_lr2.0e-4_c5p5_STW_adaptor_multi_124_resume/smmnist64.yaml \
    --checkpoint ./output_model/smmnist64_DM_Batch32_lr2.0e-4_c5p5_STW_adaptor_multi_124_resume/flowdiff_best.pth \
    --log_dir ./logs_validation/diffusion/smmnist64_DM_Batch32_lr2.0e-4_c5p5_STW_adaptor_multi_124_resume \
    --random_time False \
    --random-seed 1000 \
    --device_ids "0"