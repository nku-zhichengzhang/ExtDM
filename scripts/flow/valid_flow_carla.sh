# sh ./scripts/flow/valid_flow_carla.sh

python ./scripts/flow/valid.py \
    --cond_frames 10 \
    --pred_frames 40 \
    --num_videos 100 \
    --batch_size 100 \
    --input_size 128 \
    --config_path "./config/carla128.yaml" \
    --restore_from "/home/ubuntu15/zzc/code/videoprediction/EDM/logs_training/flow/carla128_FlowAE_Batch128_lr1e-4_Region20_perspective_Max20_Seed1000_0903/snapshots/RegionMM_0128_S150000.pth" \
    --log_dir "./logs_validation/flow/carla128" \
    --data_dir "/mnt/sda/hjy/data/carla/CARLA_Town_01_h5/" \
    --data_type "test" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"