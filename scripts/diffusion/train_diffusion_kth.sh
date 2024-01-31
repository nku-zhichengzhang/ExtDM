# sh ./scripts/diffusion/train_diffusion_kth.sh

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
AE_NAME=kth64_FlowAE_Batch128_lr1e-4_Region20_affine_scale1.00_resume
        # kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max40_2
        # kth64_FlowAE_Batch128_lr1e-4_Region20_affine_scale1.00_resume
AE_STEP=RegionMM_best_157.143
        # RegionMM_0256_S220000
        # RegionMM_best_157.143

# 从头训练
python ./scripts/diffusion/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT/KTH/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/kth64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --postfix DM_Batch8_lr5e-5_c10p20_STW_adaptor_scale1.00_multi_traj_ada

# --flowae_checkpoint $FLOWCKPT/kth64_FlowAE_Batch128_lr1e-4_Region20_affine_scale1.00_resume/snapshots/RegionMM_best_157.143.pth \
# --flowae_checkpoint $FLOWCKPT/kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max40_2/snapshots/RegionMM_0256_S220000.pth \

# 预训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/kth64/snapshots/RegionMM.pth \
#     --checkpoint ./logs_training/diffusion/kth64_0721_new/snapshots/flowdiff.pth \
#     --config ./config/kth64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix 0721_new \
#     --set-start True
