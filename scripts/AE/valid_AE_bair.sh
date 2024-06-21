# sh ./scripts/AE/valid_AE_bair.sh

data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5
pretrained_path=/home/ubuntu/zzc/data/video_prediction/AE_pretrained

python ./scripts/AE/valid.py \
    --cond_frames 2 \
    --pred_frames 28 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 64 \
    --log_dir "./logs_validation/AE/BAIR/BAIR_test" \
    --data_dir $data_path/bair_h5 \
    --config_path "$pretrained_path/BAIR/bair64_scale0.50/bair64.yaml" \
    --restore_from $pretrained_path/BAIR/bair64_scale0.50/snapshots/RegionMM.pth \
    --data_type "test" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"