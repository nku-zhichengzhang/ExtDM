# sh ./scripts/AE/valid_AE_smmnist.sh

data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5
pretrained_path=/home/ubuntu/zzc/data/video_prediction/AE_pretrained

python ./scripts/AE/valid.py \
    --cond_frames 1 \
    --pred_frames 10 \
    --num_videos 256 \
    --batch_size 2 \
    --input_size 64 \
    --log_dir "./logs_validation/AE/SMMNIST/SMMNIST_test" \
    --data_dir $data_path/smmnist_h5 \
    --config_path "$pretrained_path/SMMNIST/smmnist64_FlowAE_Batch100_lr2e-4_Region10_affine_scale0.50/smmnist64.yaml" \
    --restore_from $pretrained_path/SMMNIST/smmnist64_FlowAE_Batch100_lr2e-4_Region10_affine_scale0.50/snapshots/RegionMM.pth \
    --data_type "test" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"