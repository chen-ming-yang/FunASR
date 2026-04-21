"""
Inference the SeverityScorePredictor on a scored JSONL file.

Input JSONL format (one record per line):
    {"source": "/path/to/audio.wav", "severity_score": 0.38, ...}
    (all extra fields are preserved in the output)

Usage:
    python infer_predictor.py \
        --data val_scored.jsonl \
        --model_path pretrained_predictor.pth \
        --model_id FunAudioLLM/Fun-ASR-Nano-2512 \
        --output val_predictions.jsonl

Output JSONL adds two fields to each record:
    predicted_severity  — model prediction (float, 0-1)
    severity_error      — abs(predicted - gt_severity_score), only if severity_score present
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from funasr import AutoModel
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from mose_adapter_with_severity import SeverityScorePredictor


@torch.no_grad()
def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load records
    records = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.data}")

    # Load FunASR encoder (frozen, for feature extraction)
    print(f"Loading model: {args.model_id}")
    funasr_model = AutoModel(
        model=args.model_id,
        trust_remote_code=True,
        device=str(device),
        disable_update=True,
    )
    funasr_model.model.eval()

    frontend = funasr_model.kwargs.get("frontend")
    if frontend is None:
        raise RuntimeError("AutoModel frontend not found.")

    # Load predictor
    print(f"Loading predictor from {args.model_path}")
    predictor_state = torch.load(args.model_path, map_location="cpu")
    # Infer input_dim from first weight tensor
    first_key = next(iter(predictor_state))
    input_dim = predictor_state["temporal_pool.0.weight"].shape[1]
    hidden_dim = predictor_state["temporal_pool.0.weight"].shape[0]
    print(f"  Inferred input_dim={input_dim}, hidden_dim={hidden_dim}")

    predictor = SeverityScorePredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    predictor.load_state_dict(predictor_state)
    predictor.eval()

    # Run inference
    results = []
    skipped = 0
    errors = []

    for i, rec in enumerate(records):
        audio_path = rec["source"]
        try:
            audio = load_audio_text_image_video(audio_path, fs=frontend.fs)
            feats, feats_len = extract_fbank(audio, data_type="sound", frontend=frontend)
            feats = feats.to(device)
            feats_len = feats_len.to(device)

            enc_out, enc_out_lens = funasr_model.model.encode(feats, feats_len)[:2]

            pred = predictor(enc_out, enc_out_lens).item()
        except Exception as e:
            print(f"[WARN] Skipping {audio_path}: {e}")
            skipped += 1
            continue

        out_rec = dict(rec)
        out_rec["predicted_severity"] = round(pred, 6)

        if "severity_score" in rec and rec["severity_score"] is not None:
            err = abs(pred - float(rec["severity_score"]))
            out_rec["severity_error"] = round(err, 6)
            errors.append(err)

        results.append(out_rec)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(records)} ({skipped} skipped)")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone: {len(results)} predictions written to {output_path} ({skipped} skipped)")
    if errors:
        mae = sum(errors) / len(errors)
        print(f"MAE vs gt_severity_score: {mae:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SeverityScorePredictor inference.")
    parser.add_argument("--data", required=True, help="Input JSONL (same format as val_scored.jsonl)")
    parser.add_argument("--model_path", required=True, help="Path to pretrained_predictor.pth")
    parser.add_argument("--model_id", default="FunAudioLLM/Fun-ASR-Nano-2512", help="FunASR model for encoder")
    parser.add_argument("--output", default="predictions.jsonl", help="Output JSONL path")
    args = parser.parse_args()
    infer(args)
