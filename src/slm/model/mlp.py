from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class GELUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.model_dim = cfg.model_dim
        self.hidden_dim = cfg.hidden_dim
        
        self.fc_in = nn.Linear(cfg.model_dim, cfg.hidden_dim, bias=False)
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x)))

    def count_params(self) -> int:
        return self.fc_in.weight.numel() + self.fc_out.weight.numel()

    def flops_per_token(self) -> float:
        fc_in_term =  2 * self.model_dim * self.hidden_dim
        fc_out_term = 2 * self.hidden_dim * self.model_dim
        activation = 6 * self.hidden_dim
        return fc_in_term + fc_out_term + activation

class ReLU2MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.model_dim = cfg.model_dim
        self.hidden_dim = cfg.hidden_dim
        self.fc_in = nn.Linear(cfg.model_dim, cfg.hidden_dim, bias=False)
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        return self.fc_out(F.relu(h).square())

    def count_params(self) -> int:
        return self.fc_in.weight.numel() + self.fc_out.weight.numel()

    def flops_per_token(self) -> float:
        fc_in_term  = 2 * self.model_dim * self.hidden_dim
        fc_out_term = 2 * self.hidden_dim * self.model_dim
        activation  = 2 * self.hidden_dim  # relu (1) + square (1)
        return fc_in_term + fc_out_term + activation



class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.model_dim = cfg.model_dim
        self.hidden_dim = cfg.hidden_dim
        self.fc_gate = nn.Linear(cfg.model_dim, cfg.hidden_dim, bias=False)
        self.fc_value = nn.Linear(cfg.model_dim, cfg.hidden_dim, bias=False)
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.fc_gate(x))
        value = self.fc_value(x)
        return self.fc_out(gate * value)

    def count_params(self) -> int:
        return (
            self.fc_gate.weight.numel()
            + self.fc_value.weight.numel()
            + self.fc_out.weight.numel()
        )

    def flops_per_token(self) -> float:
        fc_gate     = 2 * self.model_dim * self.hidden_dim
        silu        = 4 * self.hidden_dim
        fc_value    = 2 * self.model_dim * self.hidden_dim
        multiply    = self.hidden_dim
        fc_out      = 2 * self.hidden_dim * self.model_dim
        return fc_gate + silu + fc_value + multiply + fc_out

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