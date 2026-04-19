#!/bin/bash
# Generate training JSONL from CDSD dataset

cd "$(dirname "$0")"

# Generate severity scores for train and val
python mose/generate_severity_score.py \
    --input  dysar_data/train.jsonl \
    --output dysar_data/train_scored.jsonl \
    --num_workers 4

python mose/generate_severity_score.py \
    --input  dysar_data/val.jsonl \
    --output dysar_data/val_scored.jsonl \
    --num_workers 4
