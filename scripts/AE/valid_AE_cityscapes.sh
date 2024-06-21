# sh ./scripts/AE/valid_AE_cityscapes.sh

data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5
pretrained_path=/home/ubuntu/zzc/data/video_prediction/AE_pretrained

python ./scripts/AE/valid.py \
    --cond_frames 2 \
    --pred_frames 28 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 128 \
    --log_dir "./logs_validation/AE/Cityscapes/cityscapes_test" \
    --data_dir $data_path/cityscapes_h5 \
    --config_path "$pretrained_path/Cityscapes/cityscapes128_perspective/cityscapes128.yaml" \
    --restore_from $pretrained_path/Cityscapes/cityscapes128_perspective/snapshots/RegionMM.pth \
    --data_type "val" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"