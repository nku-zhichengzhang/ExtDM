# sh ./scripts/diffusion/valid_diffusion_kth.sh

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
AE_NAME=kth64_FlowAE_Batch256_lr2e-4_Region20_affine_Max40_2
AE_STEP=RegionMM_0256_S220000
DM_NAME=kth64_DM_Batch32_lr2e-4_c10p20_STW_adaptor_scale0.50_multi_traj_ada
DM_STEP=flowdiff_0032_S075000
SEED=3000
NUM_SAMPLE=100
NUM_BATCH_SIZE=1
# SEED=1000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=100

# kth64_DM_Batch32_lr2e-4_c10p4_STW_adaptor_scale0.50_multi_traj_ada
# flowdiff_best_355.236
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# random seed 1000
# bs200 - 40190MB

# kth64_DM_Batch32_lr2e-4_c10p5_STW_adaptor_multi_traj_ada
# flowdiff_0032_S098000
# random seed 7000
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada

# kth64_DM_Batch64_lr2e-4_c10p10_STW_adaptor_scale0.50_multi_traj_ada
# flowdiff_0064_S088000
# random seed 3000
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada

# kth64_DM_Batch32_lr2e-4_c10p20_STW_adaptor_scale0.50_multi_traj_ada
# flowdiff_0032_S075000
# random seed 3000
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada

################# abaltion ##############
# kth64_DM_Batch40_lr2e-4_c10p5_STW_adaptor_multi
# flowdiff_0040_S068000
# - VideoFlowDiffusion_multi
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi

# kth64_DM_Batch32_lr2e-4_c10p10_STW_adaptor_multi_traj_ada
# flowdiff_0032_S045000
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada

while [ $SEED -le 3000 ]
# while [ $SEED -le 10000 ]
do
    echo $SEED
    CUDA_VISIBLE_DEVICES=0 \
        python ./scripts/diffusion/valid.py \
        --num_sample_video  $NUM_SAMPLE \
        --total_pred_frames 40 \
        --num_videos        256 \
        --valid_batch_size  $NUM_BATCH_SIZE \
        --random-seed       $SEED \
        --dataset_path      $DATASET_PATH/KTH_h5 \
        --flowae_checkpoint $AE_CKPT/KTH/$AE_NAME/snapshots/$AE_STEP.pth \
        --config            $DM_CKPT/KTH/$DM_NAME/kth64.yaml \
        --checkpoint        $DM_CKPT/KTH/$DM_NAME/snapshots/$DM_STEP.pth \
        --log_dir           ./logs_validation/pretrained_diffusion/KTH/${DM_NAME}_${DM_STEP}_${SEED}_${NUM_SAMPLE}_not_random
    SEED=$((SEED + 1000))
done

# --random_time \