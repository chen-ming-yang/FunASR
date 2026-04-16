"""
Train the SeverityScorePredictor on frozen encoder outputs + GT severity scores.

Flow:
    1. Load FunASR-Nano encoder (frozen).
    2. For each audio, extract encoder_out.
    3. Train predictor: MSE(predictor(encoder_out), gt_severity_score).
    4. Save pretrained_predictor.pth.

Input: JSONL from generate_severity_score.py with fields:
    {"source": "/path/to/audio.wav", "severity_score": 0.87, ...}

Usage:
    python train_predictor.py \
        --data scored_data.jsonl \
        --model_id FunAudioLLM/Fun-ASR-Nano-2512 \
        --epochs 20 \
        --batch_size 16 \
        --lr 1e-3 \
        --output pretrained_predictor.pth
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from funasr import AutoModel
from mose_adapter_without_severity import SeverityScorePredictor


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


def extract_with_encoder(model, records, device):
    """Extract encoder outputs by calling the encoder directly."""
    import soundfile as sf

    encoder = model.model.audio_encoder if hasattr(model.model, 'audio_encoder') else model.model.encoder
    frontend = model.model.frontend if hasattr(model.model, 'frontend') else None
    encoder.eval()

    encoder_outs = []
    encoder_lens = []
    severity_scores = []
    skipped = 0

    for i, rec in enumerate(records):
        audio_path = rec["source"]
        score = rec.get("severity_score")
        if score is None:
            skipped += 1
            continue

        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            speech = torch.from_numpy(audio).unsqueeze(0).to(device)
            speech_lengths = torch.tensor([speech.shape[1]], dtype=torch.long, device=device)

            # Apply frontend (e.g. fbank) if available
            if frontend is not None:
                feats, feats_len = frontend(speech, speech_lengths)
            else:
                feats, feats_len = speech, speech_lengths

            with torch.no_grad():
                enc_out, enc_out_lens = encoder(feats, feats_len)

            encoder_outs.append(enc_out.squeeze(0).cpu())
            encoder_lens.append(enc_out_lens.item())
            severity_scores.append(float(score))

        except Exception as e:
            print(f"[WARN] Skipping {audio_path}: {e}")
            skipped += 1
            continue

        if (i + 1) % 200 == 0:
            print(f"  Extracted {i + 1}/{len(records)} ({skipped} skipped)")

    print(f"Extraction done: {len(severity_scores)} samples, {skipped} skipped")
    return encoder_outs, encoder_lens, severity_scores


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
    model = AutoModel(model=args.model_id, trust_remote_code=True, device=str(device))

    # Extract encoder outputs (frozen)
    print("Extracting encoder outputs...")
    encoder_outs, encoder_lens, severity_scores = extract_with_encoder(
        model, records, device
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(predictor.state_dict(), args.output)

    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Saved predictor to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SeverityScorePredictor.")
    parser.add_argument("--data", required=True, help="Scored JSONL from generate_severity_score.py")
    parser.add_argument("--model_id", default="FunAudioLLM/Fun-ASR-Nano-2512", help="FunASR model for encoder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--predictor_hidden", type=int, default=256)
    parser.add_argument("--predictor_dropout", type=float, default=0.2)
    parser.add_argument("--output", default="pretrained_predictor.pth", help="Output path for predictor weights")
    args = parser.parse_args()

    train(args)
