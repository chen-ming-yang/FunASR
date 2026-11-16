#!/usr/bin/env python3
"""
Benchmark inference latency for Fun-ASR-Nano.

Measures per-utterance:
  - RTF (Real-Time Factor): inference_time / audio_duration
  - End-to-end latency

Reports P50 / P90 / P99 percentiles and breakdown by audio duration bucket.
"""

import argparse
import json
import os
import re
import time
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def percentiles(values: List[float]) -> dict:
    arr = np.array(values)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def parse_test_jsonl(test_file: str, max_samples: int) -> List[Tuple[str, str]]:
    """Parse test.jsonl to extract (audio_path, reference_text) pairs."""
    pairs = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            messages = item.get("messages", [])
            audio_path = None
            ref_text = None
            for msg in messages:
                if msg["role"] == "user":
                    match = re.search(r"<\|startofspeech\|>!(.+?)<\|endofspeech\|>", msg["content"])
                    if match:
                        audio_path = match.group(1)
                elif msg["role"] == "assistant":
                    ref_text = msg["content"]
            if audio_path and os.path.exists(audio_path):
                pairs.append((audio_path, ref_text or ""))
    if max_samples > 0:
        pairs = pairs[:max_samples]
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Benchmark Fun-ASR-Nano latency")
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--test_file", type=str,
                        default="/cmy/cmy/FunASR/examples/industrial_data_pretraining/fun_asr_nano/data/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max samples to benchmark (0=all)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="eval/latency_report.json")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup inferences before timing")
    args = parser.parse_args()

    pairs = parse_test_jsonl(args.test_file, args.max_samples)
    print(f"Loaded {len(pairs)} audio paths for benchmarking")

    from funasr import AutoModel

    print(f"Loading model: {args.model} ...")
    model = AutoModel(
        model=args.model,
        device=args.device,
    )
    print("Model loaded.")

    print("Pre-loading audio files into memory...")
    audio_paths = []
    durations = []
    for path, _ in pairs:
        wav, sr = load_audio(path)
        audio_paths.append(path)
        durations.append(len(wav) / sr)

    # Warmup
    print(f"Running {args.warmup} warmup inferences...")
    for i in range(min(args.warmup, len(audio_paths))):
        model.generate(input=[audio_paths[i]], cache={}, batch_size=1, language="中文")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark: single-utterance inference
    print(f"Benchmarking {len(audio_paths)} utterances (batch_size=1)...")
    e2e_latencies = []
    rtfs = []

    for i, (audio_path, dur) in enumerate(zip(audio_paths, durations)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        model.generate(input=[audio_path], cache={}, batch_size=1, language="中文")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        latency = t_end - t_start
        rtf = latency / dur if dur > 0 else float("inf")

        e2e_latencies.append(latency)
        rtfs.append(rtf)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(audios)}] avg RTF so far: {np.mean(rtfs):.4f}")

    # Bucket analysis by audio duration
    buckets = {"<3s": [], "3-5s": [], "5-10s": [], "10-20s": [], ">20s": []}
    for dur, lat, rtf in zip(durations, e2e_latencies, rtfs):
        if dur < 3:
            buckets["<3s"].append({"latency": lat, "rtf": rtf, "duration": dur})
        elif dur < 5:
            buckets["3-5s"].append({"latency": lat, "rtf": rtf, "duration": dur})
        elif dur < 10:
            buckets["5-10s"].append({"latency": lat, "rtf": rtf, "duration": dur})
        elif dur < 20:
            buckets["10-20s"].append({"latency": lat, "rtf": rtf, "duration": dur})
        else:
            buckets[">20s"].append({"latency": lat, "rtf": rtf, "duration": dur})

    bucket_stats = {}
    for name, items in buckets.items():
        if items:
            bucket_stats[name] = {
                "count": len(items),
                "latency": percentiles([x["latency"] for x in items]),
                "rtf": percentiles([x["rtf"] for x in items]),
            }

    report = {
        "model": args.model,
        "device": args.device,
        "num_samples": len(audios),
        "total_audio_duration_sec": float(sum(durations)),
        "total_inference_time_sec": float(sum(e2e_latencies)),
        "overall_rtf": float(sum(e2e_latencies) / sum(durations)),
        "e2e_latency": percentiles(e2e_latencies),
        "rtf": percentiles(rtfs),
        "duration_buckets": bucket_stats,
    }

    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Samples: {len(audios)}, Total audio: {sum(durations):.1f}s")
    print(f"\nOverall RTF: {report['overall_rtf']:.4f}")
    print(f"  (RTF < 1.0 means real-time capable)")
    print(f"\nE2E Latency (seconds):")
    for k, v in report["e2e_latency"].items():
        print(f"  {k}: {v:.4f}s")
    print(f"\nRTF distribution:")
    for k, v in report["rtf"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nBy duration bucket:")
    for name, stats in bucket_stats.items():
        print(f"  {name} (n={stats['count']}): "
              f"RTF p50={stats['rtf']['p50']:.4f}, "
              f"latency p50={stats['latency']['p50']:.4f}s")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
