# sh ./scripts/flow/valid_flow_kth.sh

python ./scripts/flow/valid.py \
    --cond_frames 10 \
    --pred_frames 40 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 64 \
    --config_path "./config/kth64.yaml" \
    --restore_from "./logs_training/flow/kth64_test/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/kth64" \
    --data_dir "/mnt/sda/hjy/kth/processed" \
    --data_type "valid" \
    --save-video True \
    --random-seed 1234 \
    --gpu "0"