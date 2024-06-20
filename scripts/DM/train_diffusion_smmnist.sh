# sh ./scripts/diffusion/train_diffusion_smmnist.sh

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

DATASET_PATH=/home/u1120230288/zzc/data/video_prediction/dataset
SEED=1234
# AE_NAME=smmnist64_scale0.50
# AE_STEP=RegionMM
AE_NAME=smmnist64_FlowAE_Batch128_lr1e-4_Region10_affine_scale1.00
AE_STEP=RegionMM_best_2.183

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ./scripts/diffusion/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT/SMMNIST/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/smmnist64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1,2,3 \
    --postfix DM_Batch40_lr2e-4_c10p5_STW_adaptor_scale1.00_multi_1248

# --flowae_checkpoint $FLOWCKPT/smmnist64_FlowAE_Batch128_lr1e-4_Region10_affine_scale1.00/snapshots/RegionMM_best_2.183.pth \
# --postfix DM_Batch80_lr2e-4_c10p10_STW_adaptor_scale0.50_multi_1248 \
# --postfix DM_Batch40_lr2e-4_c10p10_STW_adaptor_scale1.00_multi_1248

# 预训练
# CUDA_VISIBLE_DEVICES=0,1 \
# python ./scripts/diffusion/run.py \
#     --set-start True \
#     --flowae_checkpoint $FLOWCKPT/smmnist64/snapshots/RegionMM.pth \
#     --config ./config/smmnist64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --random-seed 1234 \
#     --postfix DM_Batch32_lr2e-4_c10p5_STW_adaptor_multi_traj
#     --checkpoint ./logs_training/diffusion/smmnist64_DM_Batch32_lr2e-4_c10p5_STW_adaptor/snapshots/flowdiff.pth \




