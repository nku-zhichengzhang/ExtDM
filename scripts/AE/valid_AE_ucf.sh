# sh ./scripts/AE/valid_AE_ucf.sh

data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5
pretrained_path=/home/ubuntu/zzc/data/video_prediction/AE_pretrained

python ./scripts/AE/valid.py \
    --cond_frames 1 \
    --pred_frames 10 \
    --num_videos 256 \
    --batch_size 2 \
    --input_size 64 \
    --log_dir "./logs_validation/AE/UCF/ucf101_test" \
    --data_dir $data_path/UCF101_h5 \
    --config_path "$pretrained_path/UCF101/ucf101_64_FlowAE_Batch100_lr2e-4_Region64_scale0.5/ucf101_64.yaml" \
    --restore_from $pretrained_path/UCF101/ucf101_64_FlowAE_Batch100_lr2e-4_Region64_scale0.5/snapshots/RegionMM_0100_S120000_270.85.pth \
    --data_type "test" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"