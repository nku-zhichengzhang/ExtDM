# sh ./scripts/flow/valid_flow_kth.sh

# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/flow_pretrained # u8
FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u11
# FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u16
# FLOWCKPT=/home/u009079/zzc/data/vidp/flow_pretrained # hpc_403

python ./scripts/flow/valid.py \
    --cond_frames 10 \
    --pred_frames 40 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 64 \
    --config_path "./config/kth64.yaml" \
    --restore_from "./logs_training/flow/kth64_FlowAE_Batch256_lr2e-4_Region20_affine/snapshots/RegionMM.pth" \
    --log_dir "./logs_validation/flow/kth64_FlowAE_Batch256_lr2e-4_Region20_affine" \
    --data_dir "/mnt/sda/hjy/kth/processed" \
    --data_type "valid" \
    --save-video True \
    --random-seed 1234 \
    --gpu "0"

# --restore_from "./logs_training/flow/kth64_test/snapshots/RegionMM.pth" \
# --restore_from "./logs_training/flow_pretrained/kth64/snapshots/RegionMM_0100_S038320.pth" \
# --restore_from "./logs_training/flow_pretrained/kth64/snapshots/RegionMM.pth" \
