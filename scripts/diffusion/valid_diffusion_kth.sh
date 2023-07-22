# sh ./scripts/diffusion/valid_diffusion_kth.sh

# python ./scripts/diffusion/valid.py \
#     --random-seed 1234 \
#     --config ./config/kth64.yaml \
#     --checkpoint ./logs_training/diffusion/kth64_test/snapshots/flowdiff.pth \
#     --flowae_checkpoint ./logs_training/flow_pretrained/kth64/snapshots/RegionMM.pth \
#     --log_dir ./logs_validation/diffusion/kth64 \
#     --device_ids "0"

python ./scripts/diffusion/valid.py \
    --random-seed 1000 \
    --config ./config/kth64.yaml \
    --checkpoint ./logs_training/diffusion/kth64_not_onlyflow/snapshots/flowdiff.pth \
    --flowae_checkpoint ./logs_training/flow_pretrained/kth64/snapshots/RegionMM.pth \
    --log_dir ./logs_validation/diffusion/kth64_not_onlyflow \
    --device_ids "0"