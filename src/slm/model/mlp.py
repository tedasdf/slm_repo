from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class GELUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(cfg.model_dim, cfg.hidden_dim, bias=False)
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x)))


class ReLU2MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(cfg.model_dim, cfg.hidden_dim, bias=False)
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        return self.fc_out(F.relu(h).square())


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc_gate = nn.Linear(cfg.model_dim, cfg.hidden_dim, bias=False)
        self.fc_value = nn.Linear(cfg.model_dim, cfg.hidden_dim, bias=False)
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.fc_gate(x))
        value = self.fc_value(x)
        return self.fc_out(gate * value)


MLP_REGISTRY: dict[str, type[nn.Module]] = {
    "gelu": GELUMLP,
    "relu2": ReLU2MLP,
    "swiglu": SwiGLUMLP,
}


def build_mlp(cfg: ModelConfig) -> nn.Module:
    mlp_cls = MLP_REGISTRY.get(cfg.mlp.mlp_type)
    if mlp_cls is None:
        raise ValueError(
            f"Unknown mlp_type={cfg.mlp.mlp_type!r}. "
            f"Available: {sorted(MLP_REGISTRY.keys())}"
        )
    return mlp_cls(cfg)