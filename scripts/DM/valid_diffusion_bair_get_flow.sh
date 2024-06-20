# sh ./scripts/diffusion/valid_diffusion_bair1.sh

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
SEED=1234
AE_NAME=bair64_scale0.50
AE_STEP=RegionMM
DM_NAME=bair64_DM_Batch64_lr2e-4_c2p10_STW_adaptor_scale0.50_multi_traj_ada
        # bair64_DM_Batch64_lr2.e-4_c2p5_STW_adaptor_multi_traj_resume
        # bair64_DM_Batch66_lr2.e-4_c2p7_STW_adaptor_multi_traj_ada
        # bair64_DM_Batch64_lr2e-4_c2p10_STW_adaptor_scale0.50_multi_traj_ada
DM_STEP=flowdiff_best_239.058
        # flowdiff_0064_S190000
        # flowdiff_best
        # flowdiff_best_239.058

# MODEL
# 
# bair64_DM_Batch64_lr2.e-4_c2p5_STW_adaptor_multi_traj_resume
#  - VideoFlowDiffusion_multi_w_ref
#  - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
# 
# bair64_DM_Batch66_lr2.e-4_c2p7_STW_adaptor_multi_traj_ada
# bair64_DM_Batch64_lr2e-4_c2p10_STW_adaptor_scale0.50_multi_traj_ada
#  - VideoFlowDiffusion_multi_w_ref
#  - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada

CUDA_VISIBLE_DEVICES=0 \
    python ./scripts/diffusion/valid1.py \
    --num_sample_video  6 \
    --total_pred_frames 28 \
    --num_videos        256 \
    --valid_batch_size  16 \
    --random-seed       $SEED \
    --dataset_path      $DATASET_PATH/BAIR_h5 \
    --flowae_checkpoint $AE_CKPT/BAIR/$AE_NAME/snapshots/$AE_STEP.pth \
    --config            $DM_CKPT/BAIR/$DM_NAME/bair64.yaml \
    --checkpoint        $DM_CKPT/BAIR/$DM_NAME/snapshots/$DM_STEP.pth \
    --log_dir           ./logs_validation/pretrained_diffusion/BAIR/${DM_NAME}_${DM_STEP}_${SEED}_6_getflow \
    --device_ids        0
#     --random_time       False \