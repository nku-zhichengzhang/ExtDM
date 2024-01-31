# sh ./scripts/diffusion_valid_for_MACS/valid_diffusion_smmnist.sh

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
# AE_NAME=smmnist64_scale0.50
# AE_STEP=RegionMM
# DM_NAME=smmnist64_DM_Batch40_lr2e-4_c10p10_STW_adaptor_multi_1248
# DM_STEP=flowdiff_0040_S195000
AE_NAME=smmnist64_FlowAE_Batch128_lr1e-4_Region10_affine_scale1.00
AE_STEP=RegionMM_best_2.183
DM_NAME=smmnist64_DM_Batch40_lr2e-4_c10p5_STW_adaptor_scale1.00_multi_1248
DM_STEP=flowdiff_best_44.651
SEED=1000
NUM_SAMPLE=1
NUM_BATCH_SIZE=1

# smmnist64_DM_Batch32_lr2e-4_c10p4_STW_adaptor_scale0.50_multi_1248
# flowdiff_best_23.160
# - VideoFlowDiffusion_multi1248
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi

# smmnist64_DM_Batch32_lr2.0e-4_c5p5_STW_adaptor_multi_124_resume
# flowdiff_0036_S265000
# - VideoFlowDiffusion_multi1248
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi

# smmnist64_DM_Batch40_lr2e-4_c10p10_STW_adaptor_multi_1248
# flowdiff_0040_S195000
# - VideoFlowDiffusion_multi1248
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi

CUDA_VISIBLE_DEVICES=0 \
    python ./scripts/diffusion_valid_for_MACS/valid_for_MACS.py \
    --num_sample_video  $NUM_SAMPLE \
    --total_pred_frames 10 \
    --num_videos        1 \
    --valid_batch_size  $NUM_BATCH_SIZE \
    --random-seed       $SEED \
    --dataset_path      $DATASET_PATH/SMMNIST_h5 \
    --flowae_checkpoint $AE_CKPT/SMMNIST/$AE_NAME/snapshots/$AE_STEP.pth \
    --config            $DM_CKPT/SMMNIST/$DM_NAME/smmnist64.yaml \
    --checkpoint        $DM_CKPT/SMMNIST/$DM_NAME/snapshots/$DM_STEP.pth \
    --log_dir           ./logs_validation/pretrained_diffusion/SMMNIST/${DM_NAME}_${DM_STEP}_${SEED}_${NUM_SAMPLE}_for_MACS