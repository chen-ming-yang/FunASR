"""
Train the SeverityScorePredictor on frozen encoder outputs + GT severity scores.

Flow:
    1. Load FunASR-Nano encoder (frozen).
    2. For each audio, extract encoder_out.
    3. Train predictor: MSE(predictor(encoder_out), gt_severity_score).
    4. Save the best model to pretrained_predictor.pth and keep the best 5 checkpoints.

Input: JSONL from generate_severity_score.py with fields:
    {"source": "/path/to/audio.wav", "severity_score": 0.87, ...}

Usage:
    python train_predictor.py \
        --data scored_data.jsonl \
        --model_id FunAudioLLM/Fun-ASR-Nano-2512 \
        --epochs 20 \
        --batch_size 16 \
        --extract_batch_size 8 \
        --lr 1e-3 \
        --output pretrained_predictor.pth

This writes the current best model to --output and keeps the best five checkpoints
alongside it as files like pretrained_predictor.epoch003.loss0.123456.pth.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from funasr import AutoModel
from mose_adapter_with_severity import SeverityScorePredictor


class SeverityDataset(Dataset):
    """Dataset that stores pre-extracted encoder outputs and GT severity scores."""

    def __init__(self, encoder_outs, encoder_lens, severity_scores):
        self.encoder_outs = encoder_outs
        self.encoder_lens = encoder_lens
        self.severity_scores = severity_scores

    def __len__(self):
        return len(self.severity_scores)

    def __getitem__(self, idx):
        return self.encoder_outs[idx], self.encoder_lens[idx], self.severity_scores[idx]


def collate_fn(batch):
    """Pad encoder outputs to the same length within a batch."""
    outs, lens, scores = zip(*batch)
    max_len = max(o.shape[0] for o in outs)
    dim = outs[0].shape[1]

    padded = torch.zeros(len(outs), max_len, dim)
    lengths = torch.zeros(len(outs), dtype=torch.long)
    targets = torch.zeros(len(outs))

    for i, (o, l, s) in enumerate(zip(outs, lens, scores)):
        padded[i, :o.shape[0], :] = o
        lengths[i] = l
        targets[i] = s

    return padded, lengths, targets


@torch.no_grad()
def extract_encoder_outputs(model, records, device):
    """Run the frozen encoder on all audio files and collect outputs."""
    encoder_outs = []
    encoder_lens = []
    severity_scores = []
    skipped = 0

    model.eval()
    for i, rec in enumerate(records):
        audio_path = rec["source"]
        score = rec["severity_score"]
        if score is None:
            skipped += 1
            continue

        try:
            # Use FunASR's built-in audio loading + frontend
            result = model.generate(input=audio_path, return_raw_text=True)
            # Access cached encoder output from the last forward pass
            # Alternative: call encode directly
        except Exception as e:
            print(f"[WARN] Skipping {audio_path}: {e}")
            skipped += 1
            continue

        severity_scores.append(float(score))

        if (i + 1) % 500 == 0:
            print(f"  Extracted {i + 1}/{len(records)} ({skipped} skipped)")

    print(f"Extraction done: {len(severity_scores)} samples, {skipped} skipped")
    return encoder_outs, encoder_lens, severity_scores


def extract_with_encoder(model, records, device, extract_batch_size=8):
    """Extract encoder outputs using the model's normal frontend + encode path."""
    frontend = model.kwargs.get("frontend")
    if frontend is None:
        raise RuntimeError("AutoModel frontend is required to extract encoder features.")

    if hasattr(model.model, "eval"):
        model.model.eval()

    encoder_outs = []
    encoder_lens = []
    severity_scores = []
    skipped = 0

    def process_single_record(rec):
        audio_path = rec["source"]
        score = rec.get("severity_score")
        if score is None:
            return False

        try:
            audio = load_audio_text_image_video(audio_path, fs=frontend.fs)
            feats, feats_len = extract_fbank(audio, data_type="sound", frontend=frontend)
            feats = feats.to(device)
            feats_len = feats_len.to(device)

            with torch.no_grad():
                enc_out_res = model.model.encode(feats, feats_len)
                enc_out, enc_out_lens = enc_out_res[0], enc_out_res[1]

            encoder_outs.append(enc_out.squeeze(0).cpu())
            encoder_lens.append(enc_out_lens.item())
            severity_scores.append(float(score))
            return True

        except Exception as e:
            print(f"[WARN] Skipping {audio_path}: {e}")
            return False

    def process_batch(batch_records):
        audio_paths = [rec["source"] for rec in batch_records]
        batch_scores = [float(rec["severity_score"]) for rec in batch_records]
        audios = load_audio_text_image_video(audio_paths, fs=frontend.fs)
        feats, feats_len = extract_fbank(audios, data_type="sound", frontend=frontend)
        feats = feats.to(device)
        feats_len = feats_len.to(device)

        with torch.no_grad():
            enc_out_res = model.model.encode(feats, feats_len)
            enc_out, enc_out_lens = enc_out_res[0], enc_out_res[1]

        enc_out = enc_out.cpu()
        enc_out_lens = enc_out_lens.cpu()
        for idx, score in enumerate(batch_scores):
            enc_len = int(enc_out_lens[idx])
            encoder_outs.append(enc_out[idx, :enc_len].clone())
            encoder_lens.append(enc_len)
            severity_scores.append(score)

    batch_records = []
    processed = 0

    for rec in records:
        processed += 1
        if rec.get("severity_score") is None:
            skipped += 1
        else:
            batch_records.append(rec)

        if len(batch_records) == extract_batch_size:
            try:
                process_batch(batch_records)
            except Exception as e:
                print(f"[WARN] Batch extraction failed, falling back to per-file mode: {e}")
                for batch_rec in batch_records:
                    if not process_single_record(batch_rec):
                        skipped += 1
            batch_records = []

        if processed % 200 == 0:
            print(f"  Extracted {processed}/{len(records)} ({skipped} skipped)")

    if batch_records:
        try:
            process_batch(batch_records)
        except Exception as e:
            print(f"[WARN] Batch extraction failed, falling back to per-file mode: {e}")
            for batch_rec in batch_records:
                if not process_single_record(batch_rec):
                    skipped += 1

    print(f"Extraction done: {len(severity_scores)} samples, {skipped} skipped")
    return encoder_outs, encoder_lens, severity_scores


def update_top_checkpoints(checkpoints, predictor, epoch, avg_loss, output_path, keep_top_k=5):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_name = (
        f"{output_path.stem}.epoch{epoch:03d}.loss{avg_loss:.6f}{output_path.suffix or '.pth'}"
    )
    checkpoint_path = output_path.with_name(checkpoint_name)
    torch.save(predictor.state_dict(), checkpoint_path)

    checkpoints.append({"loss": avg_loss, "epoch": epoch, "path": checkpoint_path})
    checkpoints.sort(key=lambda item: (item["loss"], item["epoch"]))

    while len(checkpoints) > keep_top_k:
        removed = checkpoints.pop()
        if removed["path"].exists():
            removed["path"].unlink()

    best_checkpoint = checkpoints[0]
    shutil.copy2(best_checkpoint["path"], output_path)
    return best_checkpoint["loss"]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load scored data
    records = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.data}")

    # Load FunASR model (encoder only needed)
    print(f"Loading model: {args.model_id}")
    model = AutoModel(
        model=args.model_id,
        trust_remote_code=True,
        device=str(device),
        disable_update=True,
    )

    # Extract encoder outputs (frozen)
    print("Extracting encoder outputs...")
    encoder_outs, encoder_lens, severity_scores = extract_with_encoder(
        model, records, device, extract_batch_size=args.extract_batch_size
    )

    if len(severity_scores) == 0:
        print("No valid samples found. Check your data.")
        return

    # Create dataset and dataloader
    dataset = SeverityDataset(encoder_outs, encoder_lens, severity_scores)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create predictor
    encoder_dim = encoder_outs[0].shape[-1]
    print(f"Encoder dim: {encoder_dim}")

    predictor = SeverityScorePredictor(
        input_dim=encoder_dim,
        hidden_dim=args.predictor_hidden,
        dropout=args.predictor_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nTraining predictor for {args.epochs} epochs...")
    best_loss = float("inf")
    best_checkpoints = []

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        total_loss = 0.0
        num_batches = 0

        for enc_out, enc_lens, gt_scores in dataloader:
            enc_out = enc_out.to(device)
            enc_lens = enc_lens.to(device)
            gt_scores = gt_scores.to(device)

            pred_scores = predictor(enc_out, enc_lens)
            loss = criterion(pred_scores, gt_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{args.epochs} | loss: {avg_loss:.6f} | lr: {lr:.2e}")

        if len(best_checkpoints) < 5 or avg_loss < best_checkpoints[-1]["loss"]:
            best_loss = update_top_checkpoints(
                best_checkpoints,
                predictor,
                epoch,
                avg_loss,
                args.output,
                keep_top_k=5,
            )
            print(f"    Saved top-5 checkpoint set. Current best loss: {best_loss:.6f}")

    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Saved best predictor to {args.output}")
    print("Saved top checkpoints:")
    for checkpoint in best_checkpoints:
        print(f"  epoch {checkpoint['epoch']:3d} | loss: {checkpoint['loss']:.6f} | path: {checkpoint['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SeverityScorePredictor.")
    parser.add_argument("--data", required=True, help="Scored JSONL from generate_severity_score.py")
    parser.add_argument("--model_id", default="FunAudioLLM/Fun-ASR-Nano-2512", help="FunASR model for encoder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--extract_batch_size", type=int, default=8, help="Batch size for encoder feature extraction")
    parser.add_argument("--predictor_hidden", type=int, default=256)
    parser.add_argument("--predictor_dropout", type=float, default=0.2)
    parser.add_argument(
        "--output",
        default="pretrained_predictor.pth",
        help="Output path for the best predictor weights; top-5 checkpoints are saved alongside it",
    )
    args = parser.parse_args()

    train(args)
