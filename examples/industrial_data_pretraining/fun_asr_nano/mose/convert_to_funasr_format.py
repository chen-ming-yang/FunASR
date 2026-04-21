"""
Convert raw JSONL (source/target/key/speaker) to FunASR-Nano chat format.

Input format:
    {"key": "...", "source": "/path/audio.wav", "target": "transcript", "speaker": "01"}

Output format:
    {
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "语音转写，不进行文本规整：<|startofspeech|>!/path/audio.wav<|endofspeech|>"},
        {"role": "assistant", "content": "transcript"}
      ],
      "speech_length": 453,
      "text_length": 7
    }

Usage:
    python convert_to_funasr_format.py \
        --input  dysar_data/train.jsonl \
        --output data/train.jsonl

    # or batch-convert all splits at once:
    python convert_to_funasr_format.py \
        --input  dysar_data/train.jsonl dysar_data/val.jsonl dysar_data/test.jsonl \
        --output data/train.jsonl       data/val.jsonl       data/test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import soundfile as sf
    def get_speech_length(wav_path: str) -> int:
        info = sf.info(wav_path)
        duration_ms = info.frames / info.samplerate * 1000
        return max(1, int((duration_ms - 25) // 10 + 1))
except ImportError:
    try:
        import torchaudio
        def get_speech_length(wav_path: str) -> int:
            info = torchaudio.info(wav_path)
            duration_ms = info.num_frames / info.sample_rate * 1000
            return max(1, int((duration_ms - 25) // 10 + 1))
    except ImportError:
        print("[WARN] Neither soundfile nor torchaudio found. speech_length will be set to 0.")
        def get_speech_length(wav_path: str) -> int:
            return 0

SYSTEM_MSG = "You are a helpful assistant."
PROMPT = "语音转写，不进行文本规整："


def convert_record(rec: dict) -> dict | None:
    audio_path = rec.get("source", "")
    transcript = rec.get("target", "")
    if not audio_path or not transcript:
        return None

    try:
        speech_length = get_speech_length(audio_path)
    except Exception as e:
        print(f"[WARN] Cannot read {audio_path}: {e}. Setting speech_length=0.")
        speech_length = 0

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"{PROMPT}<|startofspeech|>!{audio_path}<|endofspeech|>"},
            {"role": "assistant", "content": transcript},
        ],
        "speech_length": speech_length,
        "text_length": len(transcript),
    }


def convert_file(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = skipped = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out = convert_record(rec)
            if out is None:
                skipped += 1
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            converted += 1
            if converted % 500 == 0:
                print(f"  [{input_path.name}] {converted} converted...")

    print(f"  [{input_path.name}] Done: {converted} records → {output_path}  ({skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  nargs="+", required=True, help="Input JSONL file(s)")
    parser.add_argument("--output", nargs="+", required=True, help="Output JSONL file(s), same order as --input")
    args = parser.parse_args()

    if len(args.input) != len(args.output):
        print("ERROR: --input and --output must have the same number of files.")
        sys.exit(1)

    for inp, out in zip(args.input, args.output):
        print(f"Converting {inp} → {out}")
        convert_file(inp, out)
