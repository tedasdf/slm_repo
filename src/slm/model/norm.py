from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def build_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(
        f"Unknown norm_type={norm_type!r}. Available: ['layernorm', 'rmsnorm']"
    )