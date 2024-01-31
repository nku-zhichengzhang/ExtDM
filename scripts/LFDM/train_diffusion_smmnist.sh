# sh ./scripts/LFDM/train_diffusion_smmnist.sh

FLOWCKPT=/mnt/sda/hjy/data/flow_pretrained/better

python ./scripts/LFDM/run.py \
    --flowae_checkpoint $FLOWCKPT/smmnist64/snapshots/RegionMM.pth \
    --config ./config/LFDM/smmnist64_origin.yaml \
    --log_dir ./logs_training/LFDM \
    --device_ids 0,1 \
    --postfix LFDM_Batch32_lr2e-4_c10p5

# python ./scripts/LFDM/run.py \
#     --checkpoint ./logs_training/LFDM/smmnist64_origin_LFDM_Batch32_lr2e-4_c10p5/snapshots/flowdiff.pth \
#     --flowae_checkpoint $FLOWCKPT/smmnist64/snapshots/RegionMM.pth \
#     --config ./logs_training/LFDM/smmnist64_origin_LFDM_Batch32_lr2e-4_c10p5/smmnist64_origin.yaml \
#     --log_dir ./logs_training/LFDM \
#     --device_ids 0,1 \
#     --postfix LFDM_Batch32_lr2e-4_c10p5 \
#     --random-seed 1234 \
#     --set-start True