# sh ./scripts/flow/valid_flow_smmnist.sh

python ./scripts/flow/valid.py \
    --cond_frames 10 \
    --pred_frames 10 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 64 \
    --config_path "./config/smmnist64.yaml" \
    --restore_from "./logs_training/flow_pretrained/smmnist64/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/smmnist64" \
    --data_dir "/mnt/sda/hjy/SMMNIST/SMMNIST_h5" \
    --data_type "test" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"
    
# --restore-from "/mnt/sda/hjy/training_logs/cityscapes128/snapshots/RegionMM.pth"