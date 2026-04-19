# MOSE Adapter Pipeline

## Overview

The MOSE (Mixture-of-Severity-Experts) adapter is a severity-score-guided mixture-of-experts adapter that sits between the speech encoder and the LLM in FunASR-Nano. It routes encoder outputs through multiple parallel adapters, weighted by a continuous severity score that reflects audio/transcription quality.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      JointMOSAAdapter                           │
│                                                                 │
│  encoder_out ──┬──────────────────────────► ConvDownsampler      │
│                │                            (2× Conv1d+ReLU,    │
│                │                             stride=2 each)     │
│                │                                │               │
│                │                                ▼               │
│                │                     ┌─── Adapter 0 (FFN) ───┐  │
│                │                     │    Adapter 1 (FFN)     │  │
│                │                     │    Adapter 2 (FFN)     │  │
│                │                     │    Adapter 3 (FFN)     │  │
│                │                     └────────────────────────┘  │
│                │                                │               │
│                │                                │ weighted sum  │
│                │                                │ (by router w) │
│                ▼                                ▼               │
│       SeverityScorePredictor          h_adapt = Σ wᵢ·Adapterᵢ  │
│       (pretrained, frozen)                      │               │
│                │                                │               │
│                ▼                                ▼               │
│       ContinuousScoreRouter ──── w ──►   output_proj (Linear)  │
│       (encoder pooled + score)               │                  │
│                                              ▼                  │
│                                         adapted output → LLM   │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Steps

### Step 0: Generate Training JSONL from CDSD Dataset (`tools/cdsd2jsonl.py`)

Converts the CDSD (Chinese Dysarthric Speech Database) dataset into FunASR-Nano training format.

**Dataset layout:**

```
D:\CDSD-Interspeech\after_catting\10h\
├── Audio/
│   ├── 01/   S001T001E000N00000.wav, ...
│   ├── 02/   ...
│   ├── 04/   ...
│   ├── 06/   ...
│   ├── 08/   ...
│   ├── 09/   ...
│   ├── 12/   ...
│   └── 20/   ...
└── Text/
    ├── 01_label.txt    "<utt_id> <transcript>" per line
    ├── 02_label.txt
    ├── ...
    └── 20-label.txt
```

**Usage:**

```bash
python tools/cdsd2jsonl.py \
    --data_dir "D:\CDSD-Interspeech\after_catting\10h" \
    --output_dir ./data \
    --val_ratio 0.05 \
    --prompt "语音转写："
```

**What it does:**

1. Scans all speaker folders under `Audio/` (e.g. `01`, `02`, ..., `20`).
2. Loads transcripts from the matching `XX_label.txt` or `XX-label.txt` in `Text/`.
3. Matches each `.wav` file to its transcript by utterance ID (filename stem).
4. Reads audio duration via `soundfile` to compute `speech_length = (duration_ms - 25) // 10 + 1`.
5. Randomly splits into `train.jsonl` (95%) and `val.jsonl` (5%).

**Output format** (one JSON per line, same as FunASR-Nano expects):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "语音转写：<|startofspeech|>!D:\\...\\S001T001E000N00000.wav<|endofspeech|>"},
    {"role": "assistant", "content": "今天"}
  ],
  "speech_length": 145,
  "text_length": 2
}
```

---

### Step 1: Generate Severity Scores (`generate_severity_score.py`)

Computes a continuous severity score (0–1) for each utterance using a reference ASR model (default: `whisper-large-v3`).

**Input:** JSONL file with `source` (audio path) and `target` (ground truth text).

**Score formula:**

```
severity = wer_weight × (1 - CER) + conf_weight × avg_confidence + wc_weight × word_count_ratio
```

| Component          | Description                                          | Default Weight |
|--------------------|------------------------------------------------------|----------------|
| `wer_error`        | 1 − character-level edit distance / ref length       | 0.4            |
| `avg_confidence`   | ASR model's average confidence                       | 0.3            |
| `word_count_ratio` | min(hyp/ref, ref/hyp), clamped to [0, 1]            | 0.3            |

**Usage:**

```bash
python generate_severity_score.py \
    --input /cmy/cmy/FunASR/examples/industrial_data_pretraining/fun_asr_nano/data/train.jsonl \
    --output train_scored.jsonl \
    --model openai/whisper-large-v3
```

**Output:** Original JSONL records augmented with `severity_score` and sub-component fields.

---

### Step 2: Prepare Model with MOSA Adapter (`prepare_mosa.py`)

Integrates the `JointMOSAAdapter` into a pretrained FunASR-Nano model and loads a pretrained severity predictor.

**What it does:**

1. Loads the base `FunAudioLLM/Fun-ASR-Nano-2512` model.
2. Creates a `JointMOSAAdapter` with these defaults:
   - `encoder_dim=1280`, `llm_dim` from LLM embedding
   - 4 adapters, `adapter_dim=2048`
   - Conv downsampler with kernel size 3
3. Loads pretrained severity predictor weights (`pretrained_predictor.pth`).
4. **Freezes** the severity predictor — only MOSA routing + adapters are trainable.
5. Saves the assembled model and config to `./pretrained_joint_mosa_model/`.

---

### Step 3: Core Adapter Module (`mose_adapter_without_severity.py`)

Contains all module definitions registered under `@tables.register("adapter_classes", "JointMOSAAdapter")`.

#### Sub-modules

| Module                    | Role                                                                 |
|---------------------------|----------------------------------------------------------------------|
| `Adapter`                 | Simple bottleneck FFN: Linear → ReLU → Linear (residual-free)       |
| `SeverityScorePredictor`  | **Pretrained & frozen.** Conv1d temporal pooling → 3-layer MLP regressor (sigmoid output). Only used for inference during adapter training. |
| `ContinuousScoreRouter`   | Projects severity score + mean-pooled encoder → softmax adapter weights |
| `ConvDownsampler`         | 2× (Conv1d stride-2 + ReLU) to reduce temporal resolution 4×        |

#### Forward Pass

```
1.  severity_score ← SeverityScorePredictor(encoder_out)  # pretrained, frozen, always inference
2.  w ← ContinuousScoreRouter(encoder_out, severity_score)         # [B, num_adapters]
3.  h_conv ← ConvDownsampler(encoder_out)                           # [B, T/4, D]
4.  h_adapt ← Σᵢ wᵢ · Adapterᵢ(h_conv)                            # weighted expert mix
5.  output ← output_proj(h_adapt)  if encoder_dim ≠ llm_dim         # [B, T/4, llm_dim]
```

#### Training vs Inference

| Aspect              | Predictor Pre-training (separate stage) | FunASR-Nano Adapter Training           | Inference                              |
|---------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| Severity score      | N/A (predictor is the target)          | Predicted by frozen `SeverityScorePredictor` | Predicted by frozen `SeverityScorePredictor` |
| Predictor gradients | Trainable (MSE loss vs GT scores)      | Frozen (no grad)                       | Frozen (no grad)                       |
| Trainable params    | Predictor only                         | Router + Adapters + output_proj        | All frozen at deploy                   |

#### Predictor Pre-training (separate stage)

The `SeverityScorePredictor` is trained **independently before** adapter training:

```python
predictor_loss = MSE(SeverityScorePredictor(encoder_out), gt_severity_score)
```

Once pretrained, its weights are loaded and **frozen for all subsequent stages**. During FunASR-Nano adapter training and final inference, the predictor runs in inference-only mode to supply severity scores to the router.

## Parameter Groups

| Group                 | Method                        | Trainable During Adapter Training |
|-----------------------|-------------------------------|-----------------------------------|
| Router + Adapters + Proj | `get_mosa_parameters()`    | Yes                               |
| Severity Predictor    | `get_predictor_parameters()`  | No (frozen)                       |
