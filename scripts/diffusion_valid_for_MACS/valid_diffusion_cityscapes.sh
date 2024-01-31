# sh ./scripts/diffusion_valid_for_MACS/valid_diffusion_cityscapes.sh

DM_CKPT=/home/u1120230288/zzc/data/video_prediction/ExtraDM_pretrained
        #hpc_hjy /home/u1120230288/zzc/data/video_prediction/ExtraDM_pretrained
AE_CKPT=/home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained
        #u8  /mnt/rhdd/zzc/data/video_prediction/flow_pretrained/better
        #u11 /mnt/sda/hjy/flow_pretrained 
        #u12 /mnt/sda/hjy/flow_pretrained/better
        #u15 /mnt/sda/hjy/data/flow_pretrained/better
        #u16 /mnt/sda/hjy/flow_pretrained/better
        #u22 /home/ubuntu/zzc/data/video_prediction/flow_pretrained/better
        #hpc_hjy /home/u1120230288/zzc/data/video_prediction/FlowAE_pretrained

DATASET_PATH=/home/u1120230288/zzc/data/video_prediction/dataset
AE_NAME=cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25
AE_STEP=RegionMM_0128_S150000
# DM_NAME=cityscapes128_DM_Batch40_lr1.5e-4_c2p7_STW_adaptor_scale0.25_multi_traj_ada
# DM_STEP=flowdiff_best
DM_NAME=cityscapes128_DM_Batch40_lr1.5e-4_c2p10_STW_adaptor_scale0.25_multi_traj_ada
DM_STEP=flowdiff_0040_S220000

# AE_NAME=cityscapes128_FlowAE_Batch128_lr2e-4_Region40_perspective_scale0.50
# AE_STEP=RegionMM_0128_S100000
# DM_NAME=cityscapes128_DM_Batch40_lr1.5e-4_c2p4_STW_adaptor_scale0.5_multi_traj_ada
# DM_STEP=flowdiff_best_215.517
# AE_NAME=cityscapes128_FlowAE_Batch64_lr1e-4_Region40_perspective_scale1.00
# AE_STEP=RegionMM_best_123.506
# DM_NAME=cityscapes128_DM_Batch32_lr1.2e-4_c2p4_STW_adaptor_scale1.0_multi_traj_ada
# DM_STEP=flowdiff_best_1305.887
SEED=1000
NUM_SAMPLE=1
NUM_BATCH_SIZE=1

# cityscapes128_DM_Batch32_lr1.5e-4_c2p4_STW_adaptor_scale0.25_multi_traj_ada
# flowdiff_best
# cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25
# RegionMM_0128_S150000
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22

# cityscapes128_DM_Batch40_lr1.5e-4_c2p5_STW_adaptor_scale0.25_multi_traj_ada
# flowdiff_best_181.577
# cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25
# RegionMM_0128_S150000
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22

# cityscapes128_DM_Batch40_lr1.5e-4_c2p7_STW_adaptor_scale0.25_multi_traj_ada
# flowdiff_best
# cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25
# RegionMM_0128_S150000
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22

# cityscapes128_DM_Batch40_lr1.5e-4_c2p10_STW_adaptor_scale0.25_multi_traj_ada
# flowdiff_0040_S220000
# cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25
# RegionMM_0128_S150000
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22

# *****
# cityscapes128_DM_Batch28_lr1.5e-4_c2p7_STW_adaptor_scale0.5_multi_traj_ada (region40)
# flowdiff_best
# cityscapes128_FlowAE_Batch128_lr2e-4_Region40_perspective_scale0.50
# RegionMM_0128_S100000
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22

# cityscapes128_DM_Batch40_lr1.5e-4_c2p4_STW_adaptor_scale0.5_multi_traj_ada
# flowdiff_best_215.517
# cityscapes128_FlowAE_Batch128_lr2e-4_Region40_perspective_scale0.50
# RegionMM_0128_S100000
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22

# cityscapes128_DM_Batch32_lr1.2e-4_c2p4_STW_adaptor_scale1.0_multi_traj_ada
# flowdiff_best_215.517
# cityscapes128_FlowAE_Batch64_lr1e-4_Region40_perspective_scale1.00
# RegionMM_best_123.506
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22

CUDA_VISIBLE_DEVICES=0 \
    python ./scripts/diffusion_valid_for_MACS/valid_for_MACS.py \
    --num_sample_video  $NUM_SAMPLE \
    --total_pred_frames 28 \
    --num_videos        1 \
    --valid_batch_size  $NUM_BATCH_SIZE \
    --random-seed       $SEED \
    --dataset_path      $DATASET_PATH/Cityscapes_h5 \
    --flowae_checkpoint $AE_CKPT/Cityscapes/$AE_NAME/snapshots/$AE_STEP.pth \
    --config            $DM_CKPT/Cityscapes/$DM_NAME/cityscapes128.yaml \
    --checkpoint        $DM_CKPT/Cityscapes/$DM_NAME/snapshots/$DM_STEP.pth \
    --log_dir           ./logs_validation/pretrained_diffusion/Cityscapes/${DM_NAME}_${DM_STEP}_${SEED}_${NUM_SAMPLE}_for_MACS