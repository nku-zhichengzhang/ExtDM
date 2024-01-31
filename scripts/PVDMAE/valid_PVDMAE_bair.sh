# sh ./scripts/PVDMAE/valid_PVDMAE_bair.sh

CUDA_VISIBLE_DEVICES=0 \
python ./scripts/PVDMAE/valid.py \
    --cond_frames 1 \
    --pred_frames 16 \
    --num_videos 256 \
    --batch_size 24 \
    --input_size 64 \
    --restore_from "/home/u1120230288/zzc/data/video_prediction/PVDM_pretrained/UCF101/autoencoder.pth" \
    --config_path "./config/PVDM/ucf101_64_pvdm.yaml" \
    --log_dir "./logs_validation/PVDMAE/bair" \
    --data_dir "/home/u1120230288/zzc/data/video_prediction/dataset/BAIR_h5" \
    --data_type "test" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"