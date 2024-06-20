# sh ./scripts/diffusion/train_diffusion_carla.sh

FLOWCKPT=/home/u1120230288/zzc/data/video_prediction/flow_pretrained/better
        #u8  /mnt/rhdd/zzc/data/video_prediction/flow_pretrained/better
        #u11 /mnt/sda/hjy/flow_pretrained 
        #u12 /mnt/sda/hjy/flow_pretrained/better
        #u15 /mnt/sda/hjy/data/flow_pretrained/better
        #u16 /mnt/sda/hjy/flow_pretrained/better
        #u22 /home/ubuntu/zzc/data/video_prediction/flow_pretrained/better
        #hpc_hjy /home/u1120230288/zzc/data/video_prediction/flow_pretrained/better

# 从头训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint $FLOWCKPT/carla128_FlowAE_Batch128_lr1e-4_Region20_perspective_Max20_Seed1000_0903/snapshots/RegionMM_0128_S150000.pth \
    --config ./config/carla128.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1,2,3,4 \
    --random-seed 1234 \
    --postfix DM_Batch64_lr2e-4_c10p20_STW_adaptor_scale0.25_multi_traj_ada

# 预训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/carla128_20region/snapshots/RegionMM.pth \
#     --config ./config/carla128.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --postfix not_onlyflow \
#     --set-start True