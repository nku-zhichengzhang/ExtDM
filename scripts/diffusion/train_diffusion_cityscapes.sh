# sh ./scripts/diffusion/train_diffusion_cityscapes.sh

FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/flow_pretrained # u8
# FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u16
# FLOWCKPT=/home/u009079/zzc/data/vidp/flow_pretrained # hpc_403

# 从头训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint $FLOWCKPT/cityscapes128_perspective/snapshots/RegionMM.pth \
#     --config ./config/cityscapes128.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix DM_Batch64_lr4e-4_c2p2

# 预训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint ./logs_training/flow/cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective/snapshots/RegionMM_0128_S150000.pth \
    --checkpoint ./logs_training/diffusion/cityscapes128_DM_Batch32_lr2e-4_c2p4/snapshots/flowdiff.pth \
    --config ./logs_training/diffusion/cityscapes128_DM_Batch32_lr2e-4_c2p4/cityscapes128.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --random-seed 1234 \
    --postfix DM_Batch32_lr2e-4_c2p4 \
    --set-start True