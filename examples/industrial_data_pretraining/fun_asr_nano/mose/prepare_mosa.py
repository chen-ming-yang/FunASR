import torch
from torch import nn
import json
import os
import funasr import AutoModel
from joint_mosa_clean import JointMOSAAdapter



model_id = "FunAudioLLM/Fun-ASR-Nano-2512"
model, kwargs = AutoModel.from_pretrained(model_id, trust_remote_code=True)

encoder_dim = 1280
llm_dim = model.llm.get_input_embeddings().weight.shape[-1]
print(f"Encoder dim: {encoder_dim}, LLM input dim: {llm_dim}")

# create joint adapter
model.audio_adapter = JointMOSAAdapter(encoder_dim=encoder_dim, 
                                       llm_dim=llm_dim,
                                       adapter_dim=2048,
                                       num_adapters=4,
                                       score_proj_dim=64,
                                       router_hidden=256,
                                       conv_kernel_size=3,
                                       predictor_hidden=256,
                                       predictor_dropout=0.2,)

# load pretrained adapter predictor into adapter
predictor_state = torch.load("pretrained_predictor.pth", map_location="cpu")
model.audio_adapter.severity_predictor.load_state_dict(predictor_state)


# freeze predictor
for param in model.audio.adapter.severity_predictor.parameters():
    param.requires_grad = False


# verify what is trainable
for name, param in model.audio_adapter.named_parameters():
    status = "trainable" if param.requires_grad else "frozen"
    print(f"{name}: {status}")

mosa_trainable = sum(
    p.numel() for p in model.audio_adapter.get_mosa_parameters()
)

predicted_frozen = sum(
    p.numel() for p in model.audio_adapter.severity_predictor.parameters()
)


print(f"MOSA trainable parameters: {mosa_trainable}")
print(f"Predictor frozen parameters: {predicted_frozen}")


total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_trainable}")
print(f"Total parameters: {total_params}")



# save
save_dir = './pretrained_joint_mosa_model'
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, 'model_with_pretrained_predictor.pth'))


config = kwargs.get("config", {})
config['audio_adapter'] = "JointMOSAAdapter"
config['audio_adapter_config'] = {
    "encoder_dim": encoder_dim,
    "llm_dim": llm_dim,
    "adapter_dim": 2048,
    "num_adapters": 4,
    "score_proj_dim": 64,
    "router_hidden": 256,
    "conv_kernel_size": 3,
    "predictor_hidden": 256,
    "predictor_dropout": 0.2,
    "freeze": False,
    "use_low_frame_rate": True,
}


with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print(f"Model with pretrained predictor saved to {save_dir}")



