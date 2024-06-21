# sh ./scripts/DM/valid_DM_bair.sh

AE_CKPT=/home/ubuntu/zzc/data/video_prediction/AE_pretrained
DM_CKPT=/home/ubuntu/zzc/data/video_prediction/DM_pretrained

data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5

AE_NAME=bair64_scale0.50
AE_STEP=RegionMM

########################################
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
# -------------------------------------
# pred4 - repeat1  - 256bs - 30966MB
# pred4 - repeat2  - 128bs - 25846MB
# pred4 - repeat4  -  64bs - 26376MB
# -------------------------------------
DM_architecture=VideoFlowDiffusion_multi_w_ref
Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
DM_NAME=bair64_DM_Batch64_lr2e-4_c2p4_STW_adaptor_scale0.50_multi_traj
DM_STEP=flowdiff_best_73000_315.362
SEED=1000
NUM_SAMPLE=1
NUM_BATCH_SIZE=128
########################################

########################################
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
# -------------------------------------
# pred5 - repeat1  - 256bs - 32474MB
# -------------------------------------
# DM_architecture=VideoFlowDiffusion_multi_w_ref
# Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
# DM_NAME=bair64_DM_Batch64_lr2.e-4_c2p5_STW_adaptor_multi_traj_resume
# DM_STEP=flowdiff_0064_S190000
# SEED=1000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=128
########################################

########################################
#  - VideoFlowDiffusion_multi_w_ref
#  - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# -------------------------------------
# pred7 - repeat1  - 256bs - 37520MB
# -------------------------------------
# DM_architecture=VideoFlowDiffusion_multi_w_ref
# Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# DM_NAME=bair64_DM_Batch66_lr2.e-4_c2p7_STW_adaptor_multi_traj_ada
# DM_STEP=flowdiff_0066_S095000
# SEED=1000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=128
########################################

########################################
#  - VideoFlowDiffusion_multi_w_ref
#  - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# -------------------------------------
# pred10 - repeat1  - 256bs - 42684MB
# -------------------------------------
# DM_architecture=VideoFlowDiffusion_multi_w_ref
# Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# DM_NAME=bair64_DM_Batch64_lr2e-4_c2p10_STW_adaptor_scale0.50_multi_traj_ada
# DM_STEP=flowdiff_best_239.058
# SEED=1000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=64
########################################

CUDA_VISIBLE_DEVICES=0 \
python ./scripts/DM/valid.py \
    --num_sample_video  $NUM_SAMPLE \
    --total_pred_frames 28 \
    --num_videos        256 \
    --valid_batch_size  $NUM_BATCH_SIZE \
    --random-seed       $SEED \
    --DM_arch           $DM_architecture \
    --Unet3D_arch       $Unet3D_architecture \
    --dataset_path      $data_path/bair_h5 \
    --flowae_checkpoint $AE_CKPT/BAIR/$AE_NAME/snapshots/$AE_STEP.pth \
    --config            $DM_CKPT/BAIR/$DM_NAME/bair64.yaml \
    --checkpoint        $DM_CKPT/BAIR/$DM_NAME/snapshots/$DM_STEP.pth \
    --log_dir           ./logs_validation/pretrained_DM/BAIR/${DM_NAME}_${DM_STEP}_${SEED}_${NUM_SAMPLE}

