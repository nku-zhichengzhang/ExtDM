# sh ./scripts/flow/valid_flow_kth.sh

python ./scripts/flow/valid.py \
    --save-video True \
    --cond-frames 10 \
    --pred-frames 40 \
    --num-videos 256 \
    --batch-size 256 \
    --input-size 64 \
    --random-seed 1234 \
    --config-path "./config/kth64.yaml" \
    --restore-from "./logs_training/flow/kth64_test/snapshots/RegionMM.pth" \
    --log-dir "./logs_validation/flow/kth64" \
    --data-dir "/mnt/sda/hjy/kth/processed" \
    --data-type "valid" \
    --gpu "0"