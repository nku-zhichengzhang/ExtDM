# sh ./scripts/flow/valid_flow_cityscapes.sh

python ./scripts/flow/valid.py \
    --cond_frames 2 \
    --pred_frames 28 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 128 \
    --config_path "./config/cityscapes128.yaml" \
    --restore_from "./logs_training/flow_pretrained/cityscapes128_perspective/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/cityscapes128" \
    --data_dir "/mnt/sda/hjy/cityscapes/cityscapes_processed/" \
    --data_type "val" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"



# sh ./scripts/flow/valid_flow_cityscapes.sh

python ./scripts/flow/valid.py \
    --cond_frames 2 \
    --pred_frames 28 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 128 \
    --config_path "./config/cityscapes128.yaml" \
    --restore_from "./logs_training/flow_pretrained/cityscapes128_perspective/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/cityscapes128" \
    --data_dir "/mnt/sda/hjy/cityscapes/cityscapes_processed/" \
    --data_type "val" \
    --save-video True \
    --random-seed 2000 \
    --gpu "0"



# sh ./scripts/flow/valid_flow_cityscapes.sh

python ./scripts/flow/valid.py \
    --cond_frames 2 \
    --pred_frames 28 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 128 \
    --config_path "./config/cityscapes128.yaml" \
    --restore_from "./logs_training/flow_pretrained/cityscapes128_perspective/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/cityscapes128" \
    --data_dir "/mnt/sda/hjy/cityscapes/cityscapes_processed/" \
    --data_type "val" \
    --save-video True \
    --random-seed 3000 \
    --gpu "0"



# sh ./scripts/flow/valid_flow_cityscapes.sh

python ./scripts/flow/valid.py \
    --cond_frames 2 \
    --pred_frames 28 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 128 \
    --config_path "./config/cityscapes128.yaml" \
    --restore_from "./logs_training/flow_pretrained/cityscapes128_perspective/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/cityscapes128" \
    --data_dir "/mnt/sda/hjy/cityscapes/cityscapes_processed/" \
    --data_type "val" \
    --save-video True \
    --random-seed 4000 \
    --gpu "0"



# sh ./scripts/flow/valid_flow_cityscapes.sh

python ./scripts/flow/valid.py \
    --cond_frames 2 \
    --pred_frames 28 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 128 \
    --config_path "./config/cityscapes128.yaml" \
    --restore_from "./logs_training/flow_pretrained/cityscapes128_perspective/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/cityscapes128" \
    --data_dir "/mnt/sda/hjy/cityscapes/cityscapes_processed/" \
    --data_type "val" \
    --save-video True \
    --random-seed 5000 \
    --gpu "0"




