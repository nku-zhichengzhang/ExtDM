# sh ./scripts/diffusion/valid_diffusion_bair.sh

python ./scripts/diffusion/new_valid.py \
    --random-seed 1000 \
    --config ./config/new_bair64.yaml \
    --checkpoint ./logs_training/diffusion/new_bair64_0722/snapshots/flowdiff.pth \
    --flowae_checkpoint /mnt/sda/hjy/flow_pretrained/bair64/snapshots/RegionMM.pth \
    --log_dir ./logs_validation/diffusion/new_bair64_0722 \
    --device_ids "0"
    