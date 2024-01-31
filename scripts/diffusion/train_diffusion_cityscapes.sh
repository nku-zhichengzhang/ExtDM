# sh ./scripts/diffusion/train_diffusion_cityscapes.sh

DM_CKPT=/home/u1120230288/zzc/data/video_prediction/ExtraDM_pretrained
        #hpc_hjy /home/u1120230288/zzc/data/video_prediction/ExtraDM_pretrained
AE_CKPT=/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained
        #u8  /mnt/rhdd/zzc/data/video_prediction/flow_pretrained/better
        #u11 /mnt/sda/hjy/flow_pretrained 
        #u12 /mnt/sda/hjy/flow_pretrained/better
        #u15 /mnt/sda/hjy/data/flow_pretrained/better
        #u16 /mnt/sda/hjy/flow_pretrained/better
        #u22 /home/ubuntu/zzc/data/video_prediction/flow_pretrained/better
        #hpc_hjy /home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained

SEED=1234
# AE_NAME=cityscapes128_FlowAE_Batch128_lr2e-4_Region40_perspective_scale0.50
# AE_STEP=RegionMM_0128_S100000
AE_NAME=cityscapes128_FlowAE_Batch64_lr1e-4_Region40_perspective_scale1.00
AE_STEP=RegionMM_best_123.506

# CUDA_VISIBLE_DEVICES=0,1,2 \
# python ./scripts/diffusion/run.py \
#     --random-seed $SEED \
#     --flowae_checkpoint $AE_CKPT/Cityscapes/$AE_NAME/snapshots/$AE_STEP.pth \
#     --config ./config/cityscapes128.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1,2 \
#     --postfix DM_Batch40_lr1.5e-4_c2p4_STW_adaptor_scale0.5_multi_traj_ada


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ./scripts/diffusion/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT/Cityscapes/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/cityscapes128.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1,2,3,4,5,6,7 \
    --postfix DM_Batch32_lr1.2e-4_c2p4_STW_adaptor_scale1.0_multi_traj_ada