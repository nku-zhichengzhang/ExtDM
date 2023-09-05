# sh ./scripts/flow/valid_flow_carla.sh

python ./scripts/flow/valid.py \
    --cond_frames 10 \
    --pred_frames 40 \
    --num_videos 100 \
    --batch_size 100 \
    --input_size 128 \
    --config_path "./logs_training/flow/carla128_FlowAE_Batch128_lr1e-4_Region40_perspective/carla128.yaml" \
    --restore_from "./logs_training/flow/carla128_FlowAE_Batch128_lr1e-4_Region40_perspective/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/carla128_FlowAE_Batch128_lr1e-4_Region40_perspective" \
    --data_dir "/mnt/sda/hjy/fdm/CARLA_Town_01_h5" \
    --data_type "test" \
    --save-video True \
    --random-seed 1234 \
    --gpu "0"