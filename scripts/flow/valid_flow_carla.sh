# sh ./scripts/flow/valid_flow_carla.sh

python ./scripts/flow/valid.py \
    --cond_frames 10 \
    --pred_frames 40 \
    --num_videos 100 \
    --batch_size 100 \
    --input_size 128 \
    --config_path "./config/carla128.yaml" \
    --restore_from "/mnt/sda/hjy/training_logs/cityscapes128/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/carla128" \
    --data_dir "/mnt/sda/hjy/fdm/CARLA_Town_01_h5" \
    --data_type "test" \ 
    --save-video True \
    --random-seed 1234 \
    --gpu "0"