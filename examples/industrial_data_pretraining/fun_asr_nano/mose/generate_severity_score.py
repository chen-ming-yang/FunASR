import json
import os
import math
import argparse
from tqdm import tqdm

import whisper


def compute_wer(reference: str, hypothesis: str) -> float:
    """Character-level WER using edit distance (works well for Chinese)."""
    ref = list(reference.replace(" ", ""))
    hyp = list(hypothesis.replace(" ", ""))
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    # Dynamic programming edit distance
    dp = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref):
        new_dp = [i + 1] + [0] * len(hyp)
        for j, hc in enumerate(hyp):
            if rc == hc:
                new_dp[j + 1] = dp[j]
            else:
                new_dp[j + 1] = 1 + min(dp[j], dp[j + 1], new_dp[j])
        dp = new_dp
    return dp[-1] / len(ref)

class ContinuousSeverityLabeler:
    def __init__(self, asr_model_id: str="openai/whisper-large-v3"):
        self.model = AutoModel(model=asr_model_id, trust_remote_code=True)
    
    def compute_severity_score(self, 
                               audio_path: str,
                               ground_truth: str,
                               wer_weight: float = 0.4,
                               conf_weight: float = 0.3,
                               wc_weight: float = 0.3) -> dict:

        result = self.model.transcribe(audio_path, language="zh")
        hypothesis = result["text"].strip()

        # Extract avg_logprob from segments → convert to confidence
        segments = result.get("segments", [])
        if segments:
            avg_logprob = sum(s["avg_logprob"] for s in segments) / len(segments)
            avg_confidence = min(math.exp(avg_logprob), 1.0)
        else:
            avg_confidence = 0.5


        try:
            utterence_wer = compute_wer(ground_truth, hypothesis)
            utterence_wer = min(utterence_wer, 1.0)
        except Exception as e:
            print(f"Error computing WER: {e}")
            utterence_wer = 1.0
        
        wer_error = 1.0 - utterence_wer

        gt_chars = list(ground_truth.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))
        if len(gt_chars) > 0:
            ratio = len(hyp_chars) / len(gt_chars)
            word_count_ratio = min(ratio, 1.0 / max(ratio, 1e-6))
            word_count_ratio = min(word_count_ratio, 1.0)
        else:
            word_count_ratio = 0.0

        combined = (
            wer_weight * wer_error +
            conf_weight * avg_confidence +
            wc_weight * word_count_ratio
        )
        combined = max(0.0, min(combined, 1.0))
        return {
            "hypothesis": hypothesis,
            "wer_error": wer_error,
            "avg_confidence": avg_confidence,
            "word_count_ratio": word_count_ratio,
            "combined_score": combined,
            "severity_score": round(combined, 4),
        }

def generate_scores_for_dataset(
    input_jsonl: str,
    output_jsonl: str,
    model_size: str = "large-v3",
    wer_weight: float = 0.4,
    conf_weight: float = 0.3,
    wc_weight: float = 0.3,
) -> None:
    """Read a JSONL file, compute severity scores for each entry, and write results.

    Supports two input formats:

    1. Simple format (one JSON object per line)::

        {"key": "...", "source": "/path/to/audio.wav", "target": "text"}

    2. FunASR-Nano chat format (from cdsd2jsonl.py)::

        {"messages": [{"role": "user", "content": "...!<audio_path>..."}, {"role": "assistant", "content": "text"}], ...}
    """
    labeler = ContinuousSeverityLabeler(model_size=model_size)

    # Load all records
    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for record in tqdm(records, desc="Scoring"):
            # Parse audio_path and ground_truth from either format
            if "source" in record:
                audio_path = record["source"]
                ground_truth = record.get("target", "")
            elif "messages" in record:
                # FunASR-Nano chat format: extract audio path from user message,
                # transcript from assistant message
                user_msg = next(m["content"] for m in record["messages"] if m["role"] == "user")
                # Audio path is between "!" and "<|endofspeech|>"
                audio_path = user_msg.split("!", 1)[1].split("<|endofspeech|>")[0]
                ground_truth = next(m["content"] for m in record["messages"] if m["role"] == "assistant")
            else:
                print(f"[WARN] Unrecognized record format, skipping")
                continue

            key = record.get("key", "")

            try:
                scores = labeler.compute_severity_score(
                    audio_path=audio_path,
                    ground_truth=ground_truth,
                    wer_weight=wer_weight,
                    conf_weight=conf_weight,
                    wc_weight=wc_weight,
                )
                hypothesis = scores.pop("hypothesis", "")
            except Exception as e:
                print(f"[WARN] Failed for key={key}: {e}")
                scores = {
                    "wer_error": None,
                    "avg_confidence": None,
                    "word_count_ratio": None,
                    "combined_score": None,
                    "severity_score": None,
                }
                hypothesis = ""

            out_record = {
                **record,
                "hypothesis": hypothesis,
                **scores,
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Done. {len(records)} records written to {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate severity scores for an ASR dataset.")
    parser.add_argument("--input",  required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--model",  default="large-v3", help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--wer_weight",  type=float, default=0.4)
    parser.add_argument("--conf_weight", type=float, default=0.3)
    parser.add_argument("--wc_weight",   type=float, default=0.3)
    args = parser.parse_args()

    generate_scores_for_dataset(
        input_jsonl=args.input,
        output_jsonl=args.output,
        model_size=args.model,
        wer_weight=args.wer_weight,
        conf_weight=args.conf_weight,
        wc_weight=args.wc_weight,
    )