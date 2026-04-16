"""
Generate FunASR-Nano training JSONL from the CDSD dataset.

CDSD dataset layout:
    <root>/
        Audio/
            01/  S001T001E000N00000.wav, ...
            02/  ...
            20/  ...
        Text/
            01_label.txt   (or 20-label.txt)
            02_label.txt
            ...

Label format (space-separated):
    <utt_id> <transcript>

Usage:
    python cdsd2jsonl.py \
        --data_dir D:\CDSD-Interspeech\after_catting\10h \
        --output_dir ./data \
        --val_ratio 0.05 \
        --prompt "语音转写："
"""

import argparse
import json
import os
import random
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def find_label_file(text_dir: Path, speaker_id: str) -> Path | None:
    """Find the label file for a speaker, handling inconsistent naming (underscore vs dash)."""
    for pattern in [f"{speaker_id}_label.txt", f"{speaker_id}-label.txt"]:
        p = text_dir / pattern
        if p.exists():
            return p
    return None


def load_labels(label_path: Path) -> dict[str, str]:
    """Load utt_id -> transcript mapping from a label file."""
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                labels[utt_id] = text
            elif len(parts) == 1:
                # empty transcript, skip
                continue
    return labels


def build_record(wav_path: str, transcript: str, prompt: str) -> dict | None:
    """Build a single JSONL record matching FunASR-Nano format."""
    try:
        info = sf.info(wav_path)
        duration_ms = info.duration * 1000
        speech_length = int((duration_ms - 25) // 10 + 1)
    except Exception as e:
        print(f"[WARN] Cannot read {wav_path}: {e}")
        return None

    if speech_length <= 0:
        return None

    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"{prompt}<|startofspeech|>!{wav_path}<|endofspeech|>",
            },
            {"role": "assistant", "content": transcript},
        ],
        "speech_length": speech_length,
        "text_length": len(transcript),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert CDSD dataset to FunASR-Nano JSONL.")
    parser.add_argument("--data_dir", required=True, help="Root of CDSD dataset (contains Audio/ and Text/)")
    parser.add_argument("--output_dir", required=True, help="Directory to write train.jsonl and val.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction of data for validation (default: 0.05)")
    parser.add_argument("--prompt", default="语音转写：", help="Prompt prefix for the user message")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    audio_dir = data_dir / "Audio"
    text_dir = data_dir / "Text"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker_ids = sorted([d.name for d in audio_dir.iterdir() if d.is_dir()])
    print(f"Found {len(speaker_ids)} speakers: {speaker_ids}")

    all_records = []
    skipped = 0

    for spk_id in speaker_ids:
        label_file = find_label_file(text_dir, spk_id)
        if label_file is None:
            print(f"[WARN] No label file found for speaker {spk_id}, skipping")
            continue

        labels = load_labels(label_file)
        audio_folder = audio_dir / spk_id

        wav_files = sorted(audio_folder.glob("*.wav"))
        for wav_path in tqdm(wav_files, desc=f"Speaker {spk_id}", leave=False):
            utt_id = wav_path.stem
            if utt_id not in labels:
                skipped += 1
                continue

            transcript = labels[utt_id]
            record = build_record(str(wav_path), transcript, args.prompt)
            if record is not None:
                all_records.append(record)
            else:
                skipped += 1

    print(f"\nTotal valid records: {len(all_records)}, skipped: {skipped}")

    # Train/val split
    random.seed(args.seed)
    random.shuffle(all_records)
    val_count = max(1, int(len(all_records) * args.val_ratio))
    val_records = all_records[:val_count]
    train_records = all_records[val_count:]

    # Write JSONL
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    for path, records in [(train_path, train_records), (val_path, val_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Written {len(records)} records to {path}")


if __name__ == "__main__":
    main()
