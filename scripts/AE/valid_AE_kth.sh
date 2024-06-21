# sh ./scripts/AE/valid_AE_kth.sh

data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5
pretrained_path=/home/ubuntu/zzc/data/video_prediction/AE_pretrained

python ./scripts/AE/valid.py \
    --cond_frames 1 \
    --pred_frames 10 \
    --num_videos 256 \
    --batch_size 2 \
    --input_size 64 \
    --log_dir "./logs_validation/AE/KTH/KTH_test" \
    --data_dir $data_path/kth_h5 \
    --config_path "$pretrained_path/KTH/kth64_region10_res0.5/kth64_origin.yaml" \
    --restore_from $pretrained_path/KTH/kth64_region10_res0.5/snapshots/RegionMM.pth \
    --data_type "valid" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"