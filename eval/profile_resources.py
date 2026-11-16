#!/usr/bin/env python3
"""
Profile peak memory and GPU VRAM usage during Fun-ASR-Nano inference.

Measures:
  - Peak GPU memory allocated / reserved
  - CPU RSS (Resident Set Size) peak
  - Model size on disk
  - Per-inference memory delta
"""

import argparse
import gc
import json
import os
import re
import time
from typing import List, Tuple

import numpy as np
import psutil
import soundfile as sf
import torch


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def parse_test_jsonl(test_file: str, max_samples: int) -> List[str]:
    """Parse test.jsonl to extract audio paths."""
    paths = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            messages = item.get("messages", [])
            for msg in messages:
                if msg["role"] == "user":
                    match = re.search(r"<\|startofspeech\|>!(.+?)<\|endofspeech\|>", msg["content"])
                    if match:
                        audio_path = match.group(1)
                        if os.path.exists(audio_path):
                            paths.append(audio_path)
    if max_samples > 0:
        paths = paths[:max_samples]
    return paths


def get_model_size_mb(model_path: str) -> float:
    """Calculate total size of model files in MB."""
    total = 0
    if os.path.isdir(model_path):
        for root, dirs, files in os.walk(model_path):
            for f in files:
                if f.endswith((".safetensors", ".bin", ".pt", ".onnx")):
                    total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def get_gpu_memory_info(device_id: int = 0) -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device_id) / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved(device_id) / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device_id) / (1024 * 1024),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device_id) / (1024 * 1024),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Fun-ASR-Nano memory usage")
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--test_file", type=str,
                        default="/cmy/cmy/FunASR/examples/industrial_data_pretraining/fun_asr_nano/data/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Samples to profile (0=all)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="eval/resource_report.json")
    args = parser.parse_args()

    device_id = int(args.device.split(":")[-1]) if "cuda" in args.device else 0
    process = psutil.Process(os.getpid())

    from funasr import AutoModel

    # Try to find local model path for size calculation
    model_size_mb = 0.0
    try:
        cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models")
        model_name = args.model.replace("/", "/")
        local_path = os.path.join(cache_dir, model_name)
        if os.path.isdir(local_path):
            model_size_mb = get_model_size_mb(local_path)
    except Exception:
        pass
    print(f"Model size on disk: {model_size_mb:.1f} MB")

    # Baseline memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_id)

    cpu_rss_before = process.memory_info().rss / (1024 * 1024)
    print(f"CPU RSS before model load: {cpu_rss_before:.1f} MB")

    # Load model
    print(f"Loading model: {args.model} ...")
    model = AutoModel(
        model=args.model,
        device=args.device,
    )

    cpu_rss_after_load = process.memory_info().rss / (1024 * 1024)
    gpu_after_load = get_gpu_memory_info(device_id)
    print(f"CPU RSS after model load: {cpu_rss_after_load:.1f} MB "
          f"(delta: +{cpu_rss_after_load - cpu_rss_before:.1f} MB)")
    if gpu_after_load:
        print(f"GPU allocated after load: {gpu_after_load['allocated_mb']:.1f} MB")
        print(f"GPU reserved after load: {gpu_after_load['reserved_mb']:.1f} MB")

    # Load test audio
    audio_paths = parse_test_jsonl(args.test_file, args.max_samples)
    print(f"\nProfiling with {len(audio_paths)} audio samples...")

    # Reset peak stats before inference
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)

    # Run inference and track memory
    cpu_rss_peaks = []
    for i, audio_path in enumerate(audio_paths):
        model.generate(input=[audio_path], cache={}, batch_size=1, language="中文")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cpu_rss_peaks.append(process.memory_info().rss / (1024 * 1024))

    # Final stats
    gpu_after_infer = get_gpu_memory_info(device_id)
    cpu_rss_peak = max(cpu_rss_peaks) if cpu_rss_peaks else cpu_rss_after_load

    report = {
        "model": args.model,
        "device": args.device,
        "num_samples_profiled": len(audio_paths),
        "model_size_disk_mb": round(model_size_mb, 1),
        "cpu_memory": {
            "rss_before_load_mb": round(cpu_rss_before, 1),
            "rss_after_load_mb": round(cpu_rss_after_load, 1),
            "rss_peak_during_infer_mb": round(cpu_rss_peak, 1),
            "model_load_delta_mb": round(cpu_rss_after_load - cpu_rss_before, 1),
        },
        "gpu_memory": {},
    }

    if gpu_after_infer:
        report["gpu_memory"] = {
            "after_load_allocated_mb": round(gpu_after_load["allocated_mb"], 1),
            "after_load_reserved_mb": round(gpu_after_load["reserved_mb"], 1),
            "peak_allocated_mb": round(gpu_after_infer["max_allocated_mb"], 1),
            "peak_reserved_mb": round(gpu_after_infer["max_reserved_mb"], 1),
            "inference_peak_delta_mb": round(
                gpu_after_infer["max_allocated_mb"] - gpu_after_load["allocated_mb"], 1
            ),
        }

    print("\n" + "=" * 60)
    print("RESOURCE PROFILING RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model size on disk: {model_size_mb:.1f} MB")
    print(f"\nCPU Memory (RSS):")
    print(f"  Before load: {cpu_rss_before:.1f} MB")
    print(f"  After load:  {cpu_rss_after_load:.1f} MB (+{cpu_rss_after_load - cpu_rss_before:.1f})")
    print(f"  Peak infer:  {cpu_rss_peak:.1f} MB")
    if gpu_after_infer:
        print(f"\nGPU Memory (device {device_id}):")
        print(f"  After load (allocated): {gpu_after_load['allocated_mb']:.1f} MB")
        print(f"  Peak (allocated):       {gpu_after_infer['max_allocated_mb']:.1f} MB")
        print(f"  Peak (reserved):        {gpu_after_infer['max_reserved_mb']:.1f} MB")
        print(f"  Inference delta:        +{gpu_after_infer['max_allocated_mb'] - gpu_after_load['allocated_mb']:.1f} MB")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
