import torch
import torch.nn as nn
from funasr.register import tables


class Adapter(nn.Moudule):
    def __init__(self, hidden_dim:int, adapter_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
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
        return self.net(x).transpose(1, 2)
    

@tables.register("adapter_classes", "JointMOSAAdapter")
class JsontMOSAAdapter(nn.Moudule):
    """
    Simple design:
        Training: GT severity score -> Router
        Inference: Predicted severity score -> Router
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
                 freeze: bool = False,
                 use_low_frame_rate: bool = True,
                 **kwargs):
        super().__init__()
        self.num_adatpers = num_adapters
        
        self.severity_predictor = SeverityScorePredictor(input_dim=encoder_dim, hidden_dim=predictor_hidden, dropout=predictor_dropout)

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
                encoder_out_lens: torch.Tensor = None,
                severity_score: torch.Tensor = None):
        # get routing score
        if severity_score is not None:
            router_score = severity_score.detach()
        else:
            router_score = self.severity_predictor(encoder_out.detach, encoder_out_lens).detach()
        
        # MOSA forward
        w = self.router(encoder_out, router_score)
        h_conv = self.conv_downsampler(encoder_out)

        adapted_out_lens = encoder_out_lens
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