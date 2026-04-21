#!/bin/bash
# Convert raw JSONL splits (train/val/test) to FunASR-Nano chat format.
# Run this from the fun_asr_nano/ directory.
#
# Usage:
#   bash mose/convert_data.sh
#
# Inputs  (dysar_data/):  train.jsonl  val.jsonl  test.jsonl
# Outputs (data/):        train.jsonl  val.jsonl  test.jsonl

set -e

workspace=$(pwd)
SCRIPT="${workspace}/mose/convert_to_funasr_format.py"

INPUT_DIR="${workspace}/dysar_data"
OUTPUT_DIR="${workspace}/data"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT}" \
    --input  "${INPUT_DIR}/train.jsonl" \
             "${INPUT_DIR}/val.jsonl" \
             "${INPUT_DIR}/test.jsonl" \
    --output "${OUTPUT_DIR}/train.jsonl" \
             "${OUTPUT_DIR}/val.jsonl" \
             "${OUTPUT_DIR}/test.jsonl"

echo ""
echo "Conversion complete. Output files:"
wc -l "${OUTPUT_DIR}/train.jsonl" "${OUTPUT_DIR}/val.jsonl" "${OUTPUT_DIR}/test.jsonl"
