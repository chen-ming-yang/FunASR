#!/bin/bash
# Generate training JSONL from CDSD dataset

cd "$(dirname "$0")/.."

python tools/cdsd2jsonl.py \
    --data_dir "D:\CDSD-Interspeech\after_catting\10h" \
    --output_dir ./data \
    --val_ratio 0.05 \
    --prompt "语音转写，不进行文本规整："
