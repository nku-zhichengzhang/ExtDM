# sh ./scripts/diffusion/train_diffusion_ucf.sh

DM_CKPT=/home/ubuntu/zzc/data/video_prediction_hpc/ExtraDM_pretrained
        #u22 /home/ubuntu/zzc/data/video_prediction_hpc/ExtraDM_pretrained
        #hpc_hjy /home/u1120230288/zzc/data/video_prediction/ExtraDM_pretrained
AE_CKPT=/home/ubuntu/zzc/data/video_prediction_hpc/FlowAE_pretrained
        #u8  /mnt/rhdd/zzc/data/video_prediction/flow_pretrained/better
        #u11 /mnt/sda/hjy/flow_pretrained 
        #u12 /mnt/sda/hjy/flow_pretrained/better
        #u15 /mnt/sda/hjy/data/flow_pretrained/better
        #u16 /mnt/sda/hjy/flow_pretrained/better
        #u22 /home/ubuntu/zzc/data/video_prediction_hpc/FlowAE_pretrained
        #hpc_hjy /home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained

DATASET_PATH=/home/ubuntu/zzc/data/video_prediction/UCF101/UCF101_h5
SEED=1234
AE_NAME=ucf101_64_FlowAE_Batch100_lr2e-4_Region64_scale0.5
AE_STEP=RegionMM_0100_S120000_270.85

CUDA_VISIBLE_DEVICES=6,7 \
python ./scripts/diffusion/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT/UCF101/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/ucf101_64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --postfix DM_Batch32_lr4e-4_c4p8_STW_adaptor_scale0.50_multi_traj_ada

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




