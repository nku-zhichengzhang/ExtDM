# sh ./scripts/flow/valid_flow_ucf.sh

python ./scripts/flow/valid.py \
    --cond_frames 4 \
    --pred_frames 16 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 64 \
    --config_path "./config/ucf101_64.yaml" \
    --log_dir "./logs_validation/flow/ucf101_64" \
    --data_dir "/home/ubuntu/zzc/data/video_prediction/UCF101/UCF101_h5" \
    --restore_from "/home/ubuntu/zzc/data/video_prediction_hpc/FlowAE_pretrained/UCF101/ucf101_64_FlowAE_Batch100_lr2e-4_Region64_scale0.5/snapshots/RegionMM_0100_S120000_270.85.pth" \
    --data_type "test" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"
    
# --restore_from "/mnt/sda/hjy/training_logs/cityscapes128/snapshots/RegionMM.pth"