import os
from pathlib import Path
from tqdm import tqdm
from funasr import AutoModel

model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
audio_dir = "/cmy/after_catting/10h/Audio"
output_file = "/cmy/cmy/FunASR/finetune/asr_nano_results.txt"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

wav_paths = list(Path(audio_dir).rglob("*.wav"))
print(f"Found {len(wav_paths)} wav files.")

batch_size = 32  # Send multiple files to GPU at once

with open(output_file, "w", encoding="utf-8") as f:
    for i in tqdm(range(0, len(wav_paths), batch_size), desc="Running ASR"):
        batch_paths = [str(p) for p in wav_paths[i:i+batch_size]]
        try:
            # batch_size_s allows FunASR to dynamically pack audio by length
            res = model.generate(input=batch_paths, cache={}, batch_size_s=300, disable_update_pbar=True)
            
            for r, p in zip(res, batch_paths):
                text = r.get("text", "")
                f.write(f"{p}\t{text}\n")
            f.flush()
        except Exception as e:
            print(f"Error processing batch starting at {batch_paths[0]}: {e}")

print(f"Finished! Results saved to {output_file}")