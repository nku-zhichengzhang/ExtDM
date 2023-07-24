# sh ./scripts/diffusion/valid_diffusion_bair.sh

seed=1000

while [ $seed -le 5000 ]
do
    python ./scripts/diffusion/valid.py \
        --random-seed $seed \
        --config ./logs_training/diffusion/bair64_not_onlyflow/bair64.yaml \
        --checkpoint ./logs_training/diffusion/bair64_not_onlyflow/snapshots/flowdiff.pth \
        --flowae_checkpoint ./logs_training/flow_pretrained/bair64/snapshots/RegionMM.pth \
        --log_dir ./logs_validation/diffusion/bair64_not_onlyflow_$seed \
        --device_ids "0"
    seed=$((seed + 1000))
done
