# sh ./scripts/diffusion/valid_diffusion_bair.sh

python ./scripts/diffusion/valid.py \
    --random-seed 1000 \
    --config ./config/bair64.yaml \
    --checkpoint ./logs_training/diffusion/bair64_not_onlyflow/snapshots/flowdiff.pth \
    --flowae_checkpoint ./logs_training/flow_pretrained/bair64/snapshots/RegionMM.pth \
    --log_dir ./logs_validation/diffusion/bair64_not_onlyflow \
    --device_ids "0"
    