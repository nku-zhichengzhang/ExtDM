# sh ./scripts/flow/valid_flow_carla.sh

python ./scripts/flow/valid.py \
    --postfix test 
    --save-video True
    --cond-frames 2
    --pred-frames 28
    --num-videos 256
    --batch-size 64
    --input-size 64
    --random-seed 1234
    --restore-from "/mnt/sda/hjy/training_logs/cityscapes128/snapshots/RegionMM.pth"
    --config-path "./config/bair64.yaml"
    --root-dir "./logs_validation/flow/flowautoenc_video_carla"
    --data-dir "/mnt/sda/hjy/fdm/CARLA_Town_01_h5"
    --data-type "test"
    --gpu "0"