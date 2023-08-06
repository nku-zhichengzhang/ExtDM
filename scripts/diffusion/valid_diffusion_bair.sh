# sh ./scripts/diffusion/valid_diffusion_bair.sh

seed=1000

while [ $seed -le 5000 ]
do
    python ./scripts/diffusion/valid.py \
        --random-seed $seed \
        --config ./logs_training/diffusion/bair64_0725_cond5pred14_notonlyflow_lr2e-4_MultiStepLR/bair64.yaml \
        --checkpoint ./logs_training/diffusion/bair64_0725_cond5pred14_notonlyflow_lr2e-4_MultiStepLR/snapshots/flowdiff.pth \
        --flowae_checkpoint ./logs_training/flow_pretrained/bair64/snapshots/RegionMM.pth \
        --log_dir ./logs_validation/diffusion/bair64_0725_cond5pred14_notonlyflow_lr2e-4_MultiStepLR_$seed \
        --device_ids "0"
    seed=$((seed + 1000))
done

