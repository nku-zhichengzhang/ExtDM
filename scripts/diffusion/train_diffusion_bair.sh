# sh ./scripts/diffusion/train_diffusion_bair.sh

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
AE_NAME=bair64_FlowAE_Batch128_lr1e-4_Region20_affine_scale1.00
AE_STEP=RegionMM_best_63.302

# 从头训练
python ./scripts/diffusion/run.py \
    --random-seed $SEED \
    --flowae_checkpoint $AE_CKPT/BAIR/$AE_NAME/snapshots/$AE_STEP.pth \
    --config ./config/bair64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --random-seed 1234 \
    --postfix DM_Batch48_lr1.5e-4_c2p10_STW_adaptor_scale1.00_multi_traj_ada

# --postfix baseDM_wRefFea_wcond_woOutConv_Batch32_lr2e-4_c2p7

# 预训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint {$FLOWCKPT}/bair64/snapshots/RegionMM.pth \
#     --checkpoint ./logs_training/diffusion/kth64_test_without_rf/snapshots/flowdiff.pth \
#     --config ./config/bair64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix test_without_rf \
#     --set-start True
