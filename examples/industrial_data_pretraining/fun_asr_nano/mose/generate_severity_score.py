import json
import os
import argparse
from tqdm import tqdm

from funasr import AutoModel


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
        self.model = AutoModel(model=asr_model_id, trust_remote_code=True, hub="hf")
    
    def compute_severity_score(self, 
                               audio_path: str,
                               ground_truth: str,
                               wer_weight: float = 0.4,
                               conf_weight: float = 0.3,
                               wc_weight: float = 0.3) -> float:

        result = self.model.generate(input=audio_path, return_raw_text=True)
        hypothesis = result[0]["text"]


        try:
            utterence_wer = compute_wer(ground_truth, hypothesis)
            utterence_wer = min(utterence_wer, 1.0)
        except Exception as e:
            print(f"Error computing WER: {e}")
            utterence_wer = 1.0
        
        wer_error = 1.0 - utterence_wer

        avg_confidence = result[0].get("avg_confidence", 0.5)

        gt_words = ground_truth.split()
        hyp_words = hypothesis.split()
        if len(gt_words) > 0:
            ratio = len(hyp_words) / len(gt_words)
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
            "wer_error": wer_error,
            "avg_confidence": avg_confidence,
            "word_count_ratio": word_count_ratio,
            "combined_score": combined,
            "severity_score": round(combined, 4),
        }

def generate_scores_for_dataset(
    input_jsonl: str,
    output_jsonl: str,
    asr_model_id: str = "openai/whisper-large-v3",
    wer_weight: float = 0.4,
    conf_weight: float = 0.3,
    wc_weight: float = 0.3,
) -> None:
    """Read a JSONL file, compute severity scores for each entry, and write results.

    Input format (one JSON object per line)::

        {"key": "...", "source": "/path/to/audio.wav", "target": "text", "speaker": "..."}

    Output format adds score fields to every record::

        {"key": "...", "source": "...", "target": "...", "speaker": "...",
         "hypothesis": "...", "wer_error": 0.9, "avg_confidence": 0.8,
         "word_count_ratio": 1.0, "combined_score": 0.87, "severity_score": 0.87}
    """
    labeler = ContinuousSeverityLabeler(asr_model_id=asr_model_id)

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
            key = record.get("key", "")
            audio_path = record["source"]
            ground_truth = record.get("target", "")

            try:
                scores = labeler.compute_severity_score(
                    audio_path=audio_path,
                    ground_truth=ground_truth,
                    wer_weight=wer_weight,
                    conf_weight=conf_weight,
                    wc_weight=wc_weight,
                )
                # Also capture the ASR hypothesis for inspection
                result = labeler.model.generate(input=audio_path, return_raw_text=True)
                hypothesis = result[0]["text"]
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
    parser.add_argument("--model",  default="openai/whisper-large-v3", help="ASR model id")
    parser.add_argument("--wer_weight",  type=float, default=0.4)
    parser.add_argument("--conf_weight", type=float, default=0.3)
    parser.add_argument("--wc_weight",   type=float, default=0.3)
    args = parser.parse_args()

    generate_scores_for_dataset(
        input_jsonl=args.input,
        output_jsonl=args.output,
        asr_model_id=args.model,
        wer_weight=args.wer_weight,
        conf_weight=args.conf_weight,
        wc_weight=args.wc_weight,
    )