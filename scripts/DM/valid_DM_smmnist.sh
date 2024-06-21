# sh ./scripts/DM/valid_DM_smmnist.sh

AE_CKPT=/home/ubuntu/zzc/data/video_prediction/AE_pretrained
DM_CKPT=/home/ubuntu/zzc/data/video_prediction/DM_pretrained

data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5

AE_NAME=smmnist64_scale0.50
AE_STEP=RegionMM
DM_architecture=VideoFlowDiffusion_multi1248
Unet3D_architecture=DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi

########################################
# - VideoFlowDiffusion_multi1248
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi
# -------------------------------------
DM_NAME=smmnist64_DM_Batch32_lr2e-4_c10p4_STW_adaptor_scale0.50_multi_1248
DM_STEP=flowdiff_best_23.160
SEED=1000
NUM_SAMPLE=100
NUM_BATCH_SIZE=2
########################################

########################################
# - VideoFlowDiffusion_multi1248
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi
# -------------------------------------
# DM_NAME=smmnist64_DM_Batch32_lr2.0e-4_c5p5_STW_adaptor_multi_124_resume
# DM_STEP=flowdiff_0036_S265000
# SEED=1000
# NUM_SAMPLE=100
# NUM_BATCH_SIZE=2
########################################

########################################
# - VideoFlowDiffusion_multi1248
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi
# -------------------------------------
# DM_NAME=smmnist64_DM_Batch40_lr2e-4_c10p10_STW_adaptor_multi_1248
# DM_STEP=flowdiff_0040_S195000
# SEED=1000
# NUM_SAMPLE=100
# NUM_BATCH_SIZE=2
########################################

CUDA_VISIBLE_DEVICES=1 \
    python ./scripts/DM/valid.py \
    --num_sample_video  $NUM_SAMPLE \
    --total_pred_frames 10 \
    --num_videos        256 \
    --valid_batch_size  $NUM_BATCH_SIZE \
    --random-seed       $SEED \
    --DM_arch           $DM_architecture \
    --Unet3D_arch       $Unet3D_architecture \
    --dataset_path      $data_path/smmnist_h5 \
    --flowae_checkpoint $AE_CKPT/SMMNIST/$AE_NAME/snapshots/$AE_STEP.pth \
    --config            $DM_CKPT/SMMNIST/$DM_NAME/smmnist64.yaml \
    --checkpoint        $DM_CKPT/SMMNIST/$DM_NAME/snapshots/$DM_STEP.pth \
    --log_dir           ./logs_validation/pretrained_DM/SMMNIST/${DM_NAME}_${DM_STEP}_${SEED}_${NUM_SAMPLE}