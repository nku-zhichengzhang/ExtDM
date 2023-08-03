# CUDA_VISIBLE_DEVICES=1 sh ./scripts/diffusion/valid_diffusion_cityscapes.sh

seed=1000

while [ $seed -le 5000 ]
do
    python ./scripts/diffusion/valid.py \
        --random-seed $seed \
        --config ./config/cityscapes128.yaml \
        --checkpoint ./logs_training/diffusion/cityscapes128_not_onlyflow/snapshots/flowdiff.pth \
        --flowae_checkpoint ./logs_training/flow_pretrained/cityscapes128_perspective/snapshots/RegionMM.pth \
        --log_dir ./logs_validation/diffusion/cityscapes128_not_onlyflow_$seed \
        --device_ids "0"
    seed=$((seed + 1000))
done