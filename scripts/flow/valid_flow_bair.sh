# sh ./scripts/flow/valid_flow_bair.sh

python ./scripts/flow/valid.py \
    --cond_frames 2 \
    --pred_frames 28 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 64 \
    --config_path "./config/bair64.yaml" \
    --restore_from "./logs_training/flow_pretrained/bair64/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/bair64" \
    --data_dir "/mnt/sda/hjy/bair/mcvd-pytorch/datasets/BAIR/BAIR_h5" \
    --data_type "test" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"