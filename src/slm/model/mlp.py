import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class MLPConfig:
    mlp_type: str
    d_model: int
    dropout: float


class StandardMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)

class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        hidden_dim = 256 * ((int(8 * cfg.d_model / 3) + 256 - 1) // 256)
        self.w1 = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.w_out = nn.Linear(hidden_dim, cfg.d_model, bias=False)

    def forward(self, x):
        x = torch.nn.functional.silu(self.w1(x)) * self.w2(x)
        x = self.w_out(x)
        x = self.dropout(x)
        return x
