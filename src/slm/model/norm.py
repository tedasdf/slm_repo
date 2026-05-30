from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

    def count_params(self) -> int:
        return self.weight.numel()

    def flops_per_token(self) -> float:
        mean_square = self.dim + (self.dim - 1) + 1  # dim squares + (dim-1) adds + 1 divide
        rsqrt = 1 + 1                                 # add epsilon + rsqrt
        scale_per_element = self.dim + self.dim       # multiply by rms + multiply by gamma
        return float(mean_square + rsqrt + scale_per_element)


class LayerNorm(nn.LayerNorm):
    """nn.LayerNorm with resource accounting methods."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim, eps=eps)
        self.dim = dim

    def count_params(self) -> int:
        return self.weight.numel() + self.bias.numel()

    def flops_per_token(self) -> float:
        #   mean:      (dim-1) adds + 1 divide
        #   variance:  dim subtracts + dim squares + (dim-1) adds + 1 divide + 1 add (eps) + 1 sqrt
        #   normalize: dim divides
        #   affine:    dim multiplies (gamma) + dim adds (beta)
        mean_term     = (self.dim - 1) + 1                              # sum all + divide
        variance_term = self.dim + self.dim + (self.dim - 1) + 1  # sub mean + square + sum + divide
        normal_term   = self.dim + self.dim + 1 + 1               # sub mean + divide + eps + sqrt
        affine        = self.dim + self.dim                        # gamma mul + beta add
        return mean_term + variance_term + normal_term + affine


def build_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if norm_type == "layernorm":
        return LayerNorm(dim, eps=eps)
    raise ValueError(
        f"Unknown norm_type={norm_type!r}. Available: ['layernorm', 'rmsnorm']"
    )