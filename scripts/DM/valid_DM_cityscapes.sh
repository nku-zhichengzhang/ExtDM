# sh ./scripts/DM/valid_DM_cityscapes.sh

AE_CKPT=/home/ubuntu/zzc/data/video_prediction/AE_pretrained
DM_CKPT=/home/ubuntu/zzc/data/video_prediction/DM_pretrained

data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5

DM_architecture=VideoFlowDiffusion_multi_w_ref_u22
Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22

########################################
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22
# -------------------------------------
# AE_NAME=cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25
# AE_STEP=RegionMM_0128_S150000
# DM_NAME=cityscapes128_DM_Batch32_lr1.5e-4_c2p4_STW_adaptor_scale0.25_multi_traj_ada
# DM_STEP=flowdiff_best
# SEED=1000
# NUM_SAMPLE=4
# NUM_BATCH_SIZE=32
########################################

########################################
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22
# -------------------------------------
AE_NAME=cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25
AE_STEP=RegionMM_0128_S150000
DM_NAME=cityscapes128_DM_Batch40_lr1.5e-4_c2p5_STW_adaptor_scale0.25_multi_traj_ada
DM_STEP=flowdiff_best_33000_181.577
SEED=1000
NUM_SAMPLE=100
NUM_BATCH_SIZE=128
########################################

########################################
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22
# -------------------------------------
# AE_NAME=cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25
# AE_STEP=RegionMM_0128_S150000
# DM_NAME=cityscapes128_DM_Batch40_lr1.5e-4_c2p7_STW_adaptor_scale0.25_multi_traj_ada
# DM_STEP=flowdiff_best
# SEED=1000
# NUM_SAMPLE=4
# NUM_BATCH_SIZE=32
########################################

########################################
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22
# -------------------------------------
# AE_NAME=cityscapes128_FlowAE_Batch128_lr2e-4_Region20_perspective_scale0.25
# AE_STEP=RegionMM_0128_S150000
# DM_NAME=cityscapes128_DM_Batch40_lr1.5e-4_c2p10_STW_adaptor_scale0.25_multi_traj_ada
# DM_STEP=flowdiff_0040_S220000
# SEED=1000
# NUM_SAMPLE=4
# NUM_BATCH_SIZE=32
########################################

########################################
# - VideoFlowDiffusion_multi_w_ref_u22
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22
# -------------------------------------
# AE_NAME=cityscapes128_FlowAE_Batch128_lr2e-4_Region40_perspective_scale0.50
# AE_STEP=RegionMM_0128_S100000
# DM_NAME=cityscapes128_DM_Batch28_lr1.5e-4_c2p7_STW_adaptor_scale0.5_multi_traj_ada
# DM_STEP=flowdiff_best
# SEED=1000
# NUM_SAMPLE=4
# NUM_BATCH_SIZE=32
########################################

CUDA_VISIBLE_DEVICES=0 \
    python ./scripts/DM/valid.py \
    --estimate_occlusion_map \
    --num_sample_video  $NUM_SAMPLE \
    --total_pred_frames 28 \
    --num_videos        256 \
    --valid_batch_size  $NUM_BATCH_SIZE \
    --random-seed       $SEED \
    --DM_arch           $DM_architecture \
    --Unet3D_arch       $Unet3D_architecture \
    --dataset_path      $data_path/cityscapes_h5 \
    --flowae_checkpoint $AE_CKPT/Cityscapes/$AE_NAME/snapshots/$AE_STEP.pth \
    --config            $DM_CKPT/Cityscapes/$DM_NAME/cityscapes128.yaml \
    --checkpoint        $DM_CKPT/Cityscapes/$DM_NAME/snapshots/$DM_STEP.pth \
    --log_dir           ./logs_validation/pretrained_DM/Cityscapes/${DM_NAME}_${DM_STEP}_${SEED}_${NUM_SAMPLE}
