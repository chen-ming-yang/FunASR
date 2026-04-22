#!/bin/bash
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

workspace=`pwd`

# PYTHONPATH must include mose/ so JointMOSAAdapter is importable via model.py
export PYTHONPATH="${workspace}/mose:${PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="0"

# ── Choose model ───────────────────────────────────────────────────────────────
# Option A: original FunASR-Nano from hub (no finetuning)
# model_dir="FunAudioLLM/Fun-ASR-Nano-2512"

# Option B: your finetuned checkpoint (uncomment to use)
# After training, FunASR saves checkpoints like:
#   outputs/model.pt         (best averaged model)
#   outputs/1epoch.pt  ...
# Set model_dir to the outputs/ folder; AutoModel will load model.pt from it.
model_dir="${workspace}/outputs"

# ── Input: one wav path per call, or a text-file list of paths, or a jsonl ────
# Examples:
#   Single file:   input="/cmy/after_catting/10h/Audio/08/S008T126E000N00082.wav"
#   File list:     input="${workspace}/data/test_wav.list"  (one wav path per line)
#   JSONL:         input="${workspace}/dysar_data/test.jsonl"
input="${workspace}/dysar_data/test.jsonl"

output_dir="${workspace}/outputs/infer_results"
mkdir -p "${output_dir}"

python -m funasr.bin.inference \
++model="${model_dir}" \
++trust_remote_code=true \
++remote_code="${workspace}/model.py" \
++input="${input}" \
++output_dir="${output_dir}" \
++device="cuda:0" \
++batch_size=1 \
++batch_size_s=300 \
++hotwords="" \
++language=zh
