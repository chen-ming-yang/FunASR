#!/usr/bin/env python3
"""
Calculate CER from FunASR inference output (text_tn) against test.jsonl references.

Usage:
    python cal_cer.py [--hyp text_tn] [--ref dysar_data/test.jsonl] [--per-utt]
"""

import argparse
import json


def edit_distance(r, h):
    """Levenshtein distance between two lists/strings."""
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[m]


def load_hyp(path):
    hyp = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            key = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            hyp[key] = text
    return hyp


def load_ref(path):
    ref = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref[obj["key"]] = obj["target"]
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp", default="text_tn", help="Inference output file (key + hyp per line)")
    parser.add_argument("--ref", default="dysar_data/test.jsonl", help="Reference JSONL with 'key' and 'target'")
    parser.add_argument("--per-utt", action="store_true", help="Also print per-utterance CER")
    args = parser.parse_args()

    hyp = load_hyp(args.hyp)
    ref = load_ref(args.ref)

    total_dist = 0
    total_len = 0
    missing = []

    rows = []
    for key, r in ref.items():
        if key not in hyp:
            missing.append(key)
            continue
        h = hyp[key]
        # Strip spaces for character-level comparison
        r_chars = list(r.replace(" ", ""))
        h_chars = list(h.replace(" ", ""))
        dist = edit_distance(r_chars, h_chars)
        ref_len = len(r_chars)
        total_dist += dist
        total_len += ref_len
        cer = dist / ref_len if ref_len > 0 else 0.0
        rows.append((key, r, h, dist, ref_len, cer))

    if args.per_utt:
        print(f"{'Key':<30} {'CER':>7}  REF -> HYP")
        print("-" * 80)
        for key, r, h, dist, ref_len, cer in rows:
            print(f"{key:<30} {cer:>6.2%}  {r!r} -> {h!r}")
        print()

    overall_cer = total_dist / total_len if total_len > 0 else 0.0
    print(f"Utterances evaluated : {len(rows)}")
    print(f"Total ref chars      : {total_len}")
    print(f"Total edit distance  : {total_dist}")
    print(f"Overall CER          : {overall_cer:.4f}  ({overall_cer:.2%})")

    if missing:
        print(f"\nWARNING: {len(missing)} keys in ref not found in hyp: {missing[:5]}{'...' if len(missing) > 5 else ''}")


if __name__ == "__main__":
    main()
