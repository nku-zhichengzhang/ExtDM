# sh ./scripts/diffusion/train_diffusion_smmnist.sh
SAVEROOT=./logs_training/diffusion

FLOWCKPT=/mnt/sda/hjy/flow_pretrained/better
        #u8  /mnt/rhdd/zzc/data/video_prediction/flow_pretrained/better
        #u11 /mnt/sda/hjy/flow_pretrained 
        #u12 /mnt/sda/hjy/flow_pretrained/better
        #u15 /mnt/sda/hjy/data/flow_pretrained/better
        #u16 /mnt/sda/hjy/flow_pretrained/better
        #u22 /home/ubuntu/zzc/data/video_prediction/flow_pretrained/better

DIFFCKPT=/mnt/sda/hjy/diff_pretrained
        #u8  /mnt/rhdd/zzc/data/video_prediction/diff_pretrained
        #u11 /mnt/sda/hjy/flow_pretrained 
        #u12 /mnt/sda/hjy/diff_pretrained
        #u15 /mnt/sda/hjy/data/diff_pretrained
        #u16 /mnt/sda/hjy/diff_pretrained
        #u22 /home/ubuntu/zzc/data/video_prediction/diff_pretrained

# 预训练
python ./scripts/diffusion/run.py \
    --set-start True \
    --flowae_checkpoint $FLOWCKPT/smmnist64/snapshots/RegionMM.pth \
    --checkpoint $DIFFCKPT/smmnist64_DM_Batch32_lr2e-4_c10p10/flowdiff_0032_S202500.pth* \
    --config ./config/smmnist64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --random-seed 1234 \
    --postfix DMjoint_Batch32_lr2e-4_c10p10