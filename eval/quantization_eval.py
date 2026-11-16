#!/usr/bin/env python3
"""
Quantization evaluation for Fun-ASR-Nano.

Compares baseline vs INT8 (bitsandbytes) vs INT4 (bitsandbytes NF4):
  - CER on test set
  - Inference speed (RTF)
  - Model memory footprint

Requires: bitsandbytes, jiwer
"""

import argparse
import gc
import json
import os
import re
import time
import unicodedata
from typing import Dict, List, Tuple

import jiwer
import numpy as np
import soundfile as sf
import torch


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def remove_punctuation(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def compute_cer(refs: List[str], hyps: List[str]) -> float:
    refs_clean = [remove_punctuation(r) for r in refs]
    hyps_clean = [remove_punctuation(h) for h in hyps]
    valid = [(r, h) for r, h in zip(refs_clean, hyps_clean) if len(r) > 0]
    if not valid:
        return 0.0
    refs_v, hyps_v = zip(*valid)
    return jiwer.cer(list(refs_v), list(hyps_v))


def parse_test_jsonl(test_file: str, max_samples: int) -> List[dict]:
    """Parse test.jsonl to extract (audio_path, reference_text) pairs."""
    items = []
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
            if audio_path and os.path.exists(audio_path) and ref_text:
                items.append({"audio_path": audio_path, "ref": ref_text})
    if max_samples > 0:
        items = items[:max_samples]
    return items


def evaluate_model(model, audio_paths, refs, durations) -> dict:
    """Run inference and compute CER + RTF."""
    hyps = []
    latencies = []

    for audio_path, dur in zip(audio_paths, durations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model.generate(input=[audio_path], cache={}, batch_size=1, language="中文")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
        hyps.append(results[0].get("text", "") if isinstance(results[0], dict) else str(results[0]))

    total_audio = sum(durations)
    total_infer = sum(latencies)
    cer = compute_cer(refs, hyps)

    return {
        "cer": round(cer * 100, 2),
        "rtf": round(total_infer / total_audio, 4),
        "total_infer_sec": round(total_infer, 2),
        "total_audio_sec": round(total_audio, 2),
        "avg_latency_sec": round(total_infer / len(audio_paths), 4),
    }


def get_gpu_mem(device_id: int = 0) -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device_id) / (1024 * 1024)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Quantization evaluation for Fun-ASR-Nano")
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--test_file", type=str,
                        default="/cmy/cmy/FunASR/examples/industrial_data_pretraining/fun_asr_nano/data/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max samples for evaluation (0=all)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="eval/quantization_report.json")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_int8", action="store_true")
    parser.add_argument("--skip_int4", action="store_true")
    args = parser.parse_args()

    device_id = int(args.device.split(":")[-1]) if "cuda" in args.device else 0

    items = parse_test_jsonl(args.test_file, args.max_samples)
    print(f"Loaded {len(items)} test samples")

    audio_paths = []
    refs = []
    durations = []
    for item in items:
        wav, sr = load_audio(item["audio_path"])
        audio_paths.append(item["audio_path"])
        refs.append(item["ref"])
        durations.append(len(wav) / sr)

    from funasr import AutoModel

    results = {}
    configs_to_run = []
    if not args.skip_baseline:
        configs_to_run.append(("baseline", "baseline"))
    if not args.skip_int8:
        configs_to_run.append(("int8", "int8"))
    if not args.skip_int4:
        configs_to_run.append(("int4_nf4", "int4"))

    for name, quant_type in configs_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_id)

        try:
            if quant_type == "baseline":
                model = AutoModel(model=args.model, device=args.device)
            elif quant_type == "int8":
                model = AutoModel(
                    model=args.model,
                    device=args.device,
                    quantize="int8",
                )
            elif quant_type == "int4":
                model = AutoModel(
                    model=args.model,
                    device=args.device,
                    quantize="int4",
                )
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            results[name] = {"error": str(e)}
            continue

        gpu_after_load = get_gpu_mem(device_id) if torch.cuda.is_available() else 0

        # Warmup
        model.generate(input=[audio_paths[0]], cache={}, batch_size=1, language="中文")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device_id)

        eval_result = evaluate_model(model, audio_paths, refs, durations)
        peak_gpu = get_gpu_mem(device_id)

        eval_result["gpu_model_mb"] = round(gpu_after_load, 1)
        eval_result["gpu_peak_mb"] = round(peak_gpu, 1)
        results[name] = eval_result

        print(f"  CER: {eval_result['cer']:.2f}%")
        print(f"  RTF: {eval_result['rtf']:.4f}")
        print(f"  GPU model: {gpu_after_load:.1f} MB, peak: {peak_gpu:.1f} MB")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "model": args.model,
        "num_samples": len(items),
        "total_audio_sec": round(sum(durations), 1),
        "results": results,
    }

    print(f"\n{'='*60}")
    print("QUANTIZATION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'CER%':<8} {'RTF':<8} {'GPU(MB)':<10}")
    print("-" * 60)

    baseline_cer = results.get("baseline", {}).get("cer", None)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<15} {'ERROR':<8}")
            continue
        cer_delta = ""
        if baseline_cer is not None and name != "baseline":
            delta = res["cer"] - baseline_cer
            cer_delta = f" ({'+' if delta > 0 else ''}{delta:.2f})"
        print(f"{name:<15} {res['cer']:<8.2f}{cer_delta} "
              f"{res['rtf']:<8.4f} {res.get('gpu_peak_mb', 'N/A'):<10}")

    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
