# sh ./scripts/diffusion/train_diffusion_bair.sh

# 从头训练
python ./scripts/diffusion/run.py \
    --flowae_checkpoint ./logs_training/flow_pretrained/bair64/snapshots/RegionMM.pth \
    --config ./config/bair64.yaml \
    --log_dir ./logs_training/diffusion \
    --device_ids 0,1 \
    --random-seed 1234 \
    --postfix 0724_cond5pred14_notonlyflow_lr2e-4_MultiStepLR

# 预训练
# python ./scripts/diffusion/run.py \
#     --flowae_checkpoint ./logs_training/flow_pretrained/bair64/snapshots/RegionMM.pth \
#     --checkpoint ./logs_training/diffusion/bair64_0724_notonlyflow_lr2e-4_MultiStepLR/snapshots/flowdiff.pth \
#     --config ./config/bair64.yaml \
#     --log_dir ./logs_training/diffusion \
#     --device_ids 0,1 \
#     --random-seed 1234 \
#     --postfix 0724_notonlyflow_lr2e-4_MultiStepLR \
#     --set-start True