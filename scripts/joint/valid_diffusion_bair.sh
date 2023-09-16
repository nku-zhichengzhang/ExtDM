# sh ./scripts/diffusion/valid_diffusion_bair.sh

# FLOWCKPT=/mnt/rhdd/zzc/data/video_prediction/flow_pretrained # u8
FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u11
# FLOWCKPT=/mnt/sda/hjy/flow_pretrained # u16
# FLOWCKPT=/home/u009079/zzc/data/vidp/flow_pretrained # hpc_403

MODELNAME=bair64_DM_Batch32_lr2e-4_c2p7

seed=1000

while [ $seed -le 1000 ]
do
    python ./scripts/diffusion/valid.py \
        --random-seed $seed \
        --log_dir ./logs_validation/diffusion/${MODELNAME}_${seed} \
        --config ./logs_training/diffusion/$MODELNAME/bair64.yaml \
        --checkpoint ./logs_training/diffusion/$MODELNAME/snapshots/flowdiff_0032_S200000.pth \
        --flowae_checkpoint $FLOWCKPT/bair64/snapshots/RegionMM.pth \
        --device_ids "0"
    seed=$((seed + 1000))
done

