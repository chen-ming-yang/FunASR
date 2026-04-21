import torch
import torch.nn as nn
from funasr.register import tables


class Adapter(nn.Module):
    def __init__(self, hidden_dim:int, adapter_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class SeverityScorePredictor(nn.Module):
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.temporal_pool = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor = None):
        if encoder_out_lens is not None:
            max_len = encoder_out.shape[1]
            mask = torch.arange(max_len, device=encoder_out.device).unsqueeze(0)
            mask = mask < encoder_out_lens.unsqueeze(1)
            encoder_out = encoder_out * mask.unsqueeze(-1).float()

        x = encoder_out.transpose(1, 2)
        x = self.temporal_pool(x)
        x = x.squeeze(-1)
        return self.regressor(x).squeeze(-1)
    


class ContinousScoreRouter(nn.Module):
    def __init__(self, encoder_dim: int = 1280, score_proj_dim: int = 64, router_hidden: int = 256,
                 num_adapters: int = 4):
        super().__init__()
        self.score_encoder = nn.Sequential(
            nn.Linear(1, score_proj_dim),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(encoder_dim + score_proj_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, num_adapters),
        )

    def forward(self, encoder_out: torch.Tensor, severity_score: torch.Tensor):
        pooled = encoder_out.mean(dim=1)
        score_embed = self.score_encoder(severity_score.unsqueeze(-1))
        combined = torch.cat([pooled, score_embed], dim=-1)
        return torch.softmax(self.net(combined), dim=-1)
    

class ConvDownsampler(nn.Module):
    def __init__(self, hidden_dim: int = 256, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=self.stride, padding=self.padding),
            nn.ReLU(),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.conv(x).transpose(1, 2)
    

@tables.register("adaptor_classes", "JointMOSAAdapter")
class JointMOSAAdapter(nn.Module):
    """
    Severity-guided Mixture-of-Adapters.
    SeverityScorePredictor is pretrained & frozen; always runs inference.
    """
    def __init__(self, 
                 encoder_dim: int = 1280,
                 llm_dim: int = 3072, 
                 adapter_dim: int = 2048,
                 num_adapters: int = 4,
                 router_hidden: int = 256,
                 conv_kernel_size: int = 3,
                 predictor_hidden: int = 256,
                 predictor_dropout: float = 0.2,
                 pretrained_predictor_path: str = None,
                 freeze: bool = False,
                 use_low_frame_rate: bool = True,
                 **kwargs):
        super().__init__()
        self.num_adatpers = num_adapters
        
        self.severity_predictor = SeverityScorePredictor(input_dim=encoder_dim, hidden_dim=predictor_hidden, dropout=predictor_dropout)

        # Load pretrained predictor weights and freeze
        if pretrained_predictor_path is not None:
            state_dict = torch.load(pretrained_predictor_path, map_location="cpu")
            self.severity_predictor.load_state_dict(state_dict)
            print(f"[JointMOSAAdapter] Loaded pretrained predictor from {pretrained_predictor_path}")
        for param in self.severity_predictor.parameters():
            param.requires_grad = False
        self.severity_predictor.eval()
        print("[JointMOSAAdapter] Severity predictor frozen.")

        self.router = ContinousScoreRouter(encoder_dim=encoder_dim, 
                                           score_proj_dim=64, 
                                           router_hidden=router_hidden, 
                                           num_adapters=num_adapters)
        
        self.conv_downsampler = ConvDownsampler(hidden_dim=encoder_dim, kernel_size=conv_kernel_size)

        self.adapters = nn.ModuleList([
            Adapter(hidden_dim=encoder_dim, adapter_dim=adapter_dim) for _ in range(num_adapters)
        ])

        self.output_proj = None
        if encoder_dim != llm_dim:
            self.output_proj = nn.Linear(encoder_dim, llm_dim)
        

    def forward(self, 
                encoder_out: torch.Tensor, 
                encoder_out_lens: torch.Tensor = None):
        # severity predictor is pretrained & frozen, inference only
        with torch.no_grad():
            router_score = self.severity_predictor(encoder_out, encoder_out_lens)
        
        # MOSA forward
        w = self.router(encoder_out, router_score)
        h_conv = self.conv_downsampler(encoder_out)

        adapted_out_lens = encoder_out_lens
        if adapted_out_lens is not None:
            for _ in range(2):
                adapted_out_lens = (adapted_out_lens 
                                    + 2 * self.conv_downsampler.padding 
                                    - self.conv_downsampler.kernel_size) // self.conv_downsampler.stride + 1
        
        # weighted sum of adapter outputs
        h_adapt = torch.zeros_like(h_conv)
        for i, adapter in enumerate(self.adapters):
            h_adapt_i = adapter(h_conv)
            h_adapt = h_adapt + w[:, i].unsqueeze(-1).unsqueeze(-1) * h_adapt_i
        
        if self.output_proj is not None:
            h_adapt = self.output_proj(h_adapt)

        return h_adapt, adapted_out_lens

    def load_pretrained_adapter(self, pretrained_state_dict: dict):
        """Initialize adapters and output_proj from a pretrained Transformer adapter.

        Maps pretrained weights:
          - linear1 (encoder_dim → ffn_dim) → each adapter's net.0 (hidden_dim → adapter_dim)
          - linear2 (ffn_dim → llm_dim)     → output_proj (encoder_dim → llm_dim)

        Args:
            pretrained_state_dict: state dict of the pretrained audio_adaptor,
                e.g. model.audio_adaptor.state_dict()
        """
        # Map linear1 → each adapter's first linear layer
        if "linear1.weight" in pretrained_state_dict:
            src_w = pretrained_state_dict["linear1.weight"]
            src_b = pretrained_state_dict["linear1.bias"]
            for i, adapter in enumerate(self.adapters):
                if adapter.net[0].weight.shape == src_w.shape:
                    adapter.net[0].weight.data.copy_(src_w)
                    adapter.net[0].bias.data.copy_(src_b)
                    print(f"[init] adapter[{i}].net.0 ← pretrained linear1")
                else:
                    print(f"[skip] adapter[{i}].net.0 shape mismatch: "
                          f"{adapter.net[0].weight.shape} vs {src_w.shape}")

        # Map linear2 → output_proj
        if "linear2.weight" in pretrained_state_dict and self.output_proj is not None:
            src_w = pretrained_state_dict["linear2.weight"]
            src_b = pretrained_state_dict["linear2.bias"]
            if self.output_proj.weight.shape == src_w.shape:
                self.output_proj.weight.data.copy_(src_w)
                self.output_proj.bias.data.copy_(src_b)
                print("[init] output_proj ← pretrained linear2")
            else:
                print(f"[skip] output_proj shape mismatch: "
                      f"{self.output_proj.weight.shape} vs {src_w.shape}")


    def compute_predictor_loss(self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor, severity_score: torch.Tensor):
        predicted = self.severity_predictor(encoder_out, encoder_out_lens)
        return nn.functional.mse_loss(predicted, severity_score)
    

    def get_mosa_parameters(self):
        params = []
        params += list(self.router.parameters())
        params += list(self.conv_downsampler.parameters())
        params += list(self.adapters.parameters())

        if self.output_proj is not None:
            params += list(self.output_proj.parameters())
        return params
    
    def get_predictor_parameters(self):
        return list(self.severity_predictor.parameters())