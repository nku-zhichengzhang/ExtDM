# sh ./scripts/diffusion/valid_diffusion_bair.sh

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
AE_NAME=bair64_scale0.50
AE_STEP=RegionMM
DM_NAME=bair64_DM_Batch64_lr2e-4_c2p10_STW_adaptor_scale0.50_multi_traj_ada
DM_STEP=flowdiff_best_239.058
# SEED=2000
# NUM_SAMPLE=100
# NUM_BATCH_SIZE=2
SEED=1000
NUM_SAMPLE=1
NUM_BATCH_SIZE=100

# bair64_DM_Batch64_lr2e-4_c2p4_STW_adaptor_scale0.50_multi_traj
# flowdiff_best_315.362
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
# seed 10000
# pred4 - repeat1  - 256bs - 30966MB - 24s
# pred4 - repeat2  - 128bs - 25846MB - 24s
# pred4 - repeat4  -  64bs - 26376MB - 24s
# pred4 - repeat10 -  32bs - MB - s

# bair64_DM_Batch64_lr2.e-4_c2p5_STW_adaptor_multi_traj_resume
# flowdiff_0064_S190000
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
# seed 9000
# pred5 - repeat1  - 256bs - 32474MB - 30s

# bair64_DM_Batch66_lr2.e-4_c2p7_STW_adaptor_multi_traj_ada
# flowdiff_0066_S095000
#  - VideoFlowDiffusion_multi_w_ref
#  - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# seed 4000
# pred7 - repeat1  - 256bs - 37520MB - 32s

# bair64_DM_Batch64_lr2e-4_c2p10_STW_adaptor_scale0.50_multi_traj_ada
# flowdiff_best_239.058
#  - VideoFlowDiffusion_multi_w_ref
#  - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# seed 2000
# pred10 - repeat1  - 256bs - 42684MB - 37s

# while [ $SEED -le 2000 ]
while [ $SEED -le 10000 ]
do
    echo $SEED
    CUDA_VISIBLE_DEVICES=3 \
        python ./scripts/diffusion/valid.py \
        --num_sample_video  $NUM_SAMPLE \
        --total_pred_frames 28 \
        --num_videos        256 \
        --valid_batch_size  $NUM_BATCH_SIZE \
        --random-seed       $SEED \
        --dataset_path      $DATASET_PATH/BAIR_h5 \
        --flowae_checkpoint $AE_CKPT/BAIR/$AE_NAME/snapshots/$AE_STEP.pth \
        --config            $DM_CKPT/BAIR/$DM_NAME/bair64.yaml \
        --checkpoint        $DM_CKPT/BAIR/$DM_NAME/snapshots/$DM_STEP.pth \
        --log_dir           ./logs_validation/pretrained_diffusion/BAIR/${DM_NAME}_${DM_STEP}_${SEED}_${NUM_SAMPLE}
    SEED=$((SEED + 1000))
done

