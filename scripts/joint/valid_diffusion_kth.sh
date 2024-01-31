# sh ./scripts/diffusion/valid_diffusion_kth.sh

# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/flow_pretrained # u8
FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u11
# FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u16
# FLOWCKPT=/home/u009079/zzc/data/vidp/flow_pretrained # hpc_403

# python ./scripts/diffusion/valid.py \
#     --random-seed 1234 \
#     --config ./config/kth64.yaml \
#     --checkpoint ./logs_training/diffusion/kth64_test/snapshots/flowdiff.pth \
#     --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/kth64/snapshots/RegionMM.pth \
#     --log_dir ./logs_validation/diffusion/kth64 \
#     --device_ids "0"

python ./scripts/diffusion/valid.py \
    --random-seed 1234 \
    --flowae_checkpoint $FLOWCKPT/kth64/snapshots/RegionMM.pth \
    --config ./logs_training/diffusion/kth64_DM_Batch32_lr2e-4_c10p5/kth64.yaml \
    --checkpoint ./logs_training/diffusion/kth64_DM_Batch32_lr2e-4_c10p5/snapshots/flowdiff.pth \
    --log_dir ./logs_validation/diffusion/kth64_DM_Batch32_lr2e-4_c10p5_0825_random \
    --device_ids "0"

# python ./scripts/diffusion/valid.py \
#     --random-seed 1234 \
#     --config ./config/kth64.yaml \
#     --checkpoint ./logs_training/diffusion/snapshots-joint-steplr-random-onlyflow-train-regionmm-temp/flowdiff.pth \
#     --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/kth64/snapshots/RegionMM.pth \
#     --log_dir ./logs_validation/diffusion/kth64_test_without_rf \
#     --device_ids "0"

# python ./scripts/diffusion/valid.py \
#     --random-seed 1234 \
#     --config ./config/kth64.yaml \
#     --checkpoint ./logs_training/diffusion/snapshots-joint-steplr-random-onlyflow-train-regionmm-temp-rf/flowdiff.pth \
#     --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/kth64/snapshots/RegionMM.pth \
#     --log_dir ./logs_validation/diffusion/kth64_test_with_rf \
#     --device_ids "0"
