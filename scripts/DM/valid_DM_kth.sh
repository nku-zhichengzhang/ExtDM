# sh ./scripts/DM/valid_DM_kth.sh

AE_CKPT=/home/ubuntu/zzc/data/video_prediction/AE_pretrained
DM_CKPT=/home/ubuntu/zzc/data/video_prediction/DM_pretrained

data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5

AE_NAME=kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max40_2
AE_STEP=RegionMM_0256_S220000

########################################
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# -------------------------------------
DM_architecture=VideoFlowDiffusion_multi_w_ref
Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
DM_NAME=kth64_DM_Batch32_lr2e-4_c10p4_STW_adaptor_scale0.50_multi_traj_ada
DM_STEP=flowdiff_best_355.236
SEED=1000
NUM_SAMPLE=100
NUM_BATCH_SIZE=100
########################################

########################################
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# -------------------------------------
# DM_architecture=VideoFlowDiffusion_multi_w_ref
# Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# DM_NAME=kth64_DM_Batch32_lr2e-4_c10p5_STW_adaptor_multi_traj_ada
# DM_STEP=flowdiff_0032_S098000
# SEED=7000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=100
########################################

########################################
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# -------------------------------------
# DM_architecture=VideoFlowDiffusion_multi_w_ref
# Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# DM_NAME=kth64_DM_Batch64_lr2e-4_c10p10_STW_adaptor_scale0.50_multi_traj_ada
# DM_STEP=flowdiff_0064_S088000
# SEED=3000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=100
########################################

########################################
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# -------------------------------------
# DM_architecture=VideoFlowDiffusion_multi_w_ref
# Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# DM_NAME=kth64_DM_Batch32_lr2e-4_c10p20_STW_adaptor_scale0.50_multi_traj_ada
# DM_STEP=flowdiff_0032_S075000
# SEED=3000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=100
########################################

CUDA_VISIBLE_DEVICES=0 \
    python ./scripts/DM/valid.py \
    --num_sample_video  $NUM_SAMPLE \
    --total_pred_frames 40 \
    --num_videos        256 \
    --valid_batch_size  $NUM_BATCH_SIZE \
    --random-seed       $SEED \
    --DM_arch           $DM_architecture \
    --Unet3D_arch       $Unet3D_architecture \
    --dataset_path      $data_path/kth_h5 \
    --flowae_checkpoint $AE_CKPT/KTH/$AE_NAME/snapshots/$AE_STEP.pth \
    --config            $DM_CKPT/KTH/$DM_NAME/kth64.yaml \
    --checkpoint        $DM_CKPT/KTH/$DM_NAME/snapshots/$DM_STEP.pth \
    --log_dir           ./logs_validation/pretrained_DM/KTH/${DM_NAME}_${DM_STEP}_${SEED}_${NUM_SAMPLE}
# --random_time \
