from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .attention import build_attention
from .config import ModelConfig
from .mlp import build_mlp
from .norm import build_norm


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block:

        x = x + Attention(Norm(x))
        x = x + MLP(Norm(x))

    Notes
    -----
    - This keeps the block simple and architecture-focused.
    - Any attention-specific kwargs (e.g. attention mask, cache) are passed
      through to the attention module via **attn_kwargs.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.attn_norm = build_norm(cfg.norm_type, cfg.model_dim, cfg.norm_eps)
        self.attn = build_attention(cfg)

        self.mlp_norm = build_norm(cfg.norm_type, cfg.model_dim, cfg.norm_eps)
        self.mlp = build_mlp(cfg)

        self.grad_norm_inspect_enabled: bool = False
        self.grad_norm_inspect_active: bool = False
        self.grad_norm_inspect_layer_idx: int | None = None
        self.last_resid_grad_norm: float | None = None

    def forward(self, x: torch.Tensor, **attn_kwargs: Any) -> torch.Tensor:
        x0 = x
        if (
            self.grad_norm_inspect_enabled
            and self.grad_norm_inspect_active
            and x0.requires_grad
        ):
            x0.register_hook(self._record_resid_grad_norm)

        attn_out = self.attn(self.attn_norm(x0), **attn_kwargs)
        x1 = x0 + attn_out

        mlp_out = self.mlp(self.mlp_norm(x1))
        return x1 + mlp_out

    def _record_resid_grad_norm(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = grad.detach().float().norm().item()
        if self.last_resid_grad_norm is None:
            self.last_resid_grad_norm = grad_norm
        else:
            self.last_resid_grad_norm = max(self.last_resid_grad_norm, grad_norm)
        return grad


    def count_params(self) -> int:
        return (
            self.attn_norm.count_params()
            + self.attn.count_params()
            + self.mlp_norm.count_params()
            + self.mlp.count_params()
        )

    def flops_per_token(self, seq_len: int) -> float:
        return (
            self.attn_norm.flops_per_token()
            + self.attn.flops_per_token(seq_len)
            + self.mlp_norm.flops_per_token()
            + self.mlp.flops_per_token()
        )


BLOCK_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline": TransformerBlock,
}


def build_block(cfg: ModelConfig) -> nn.Module:
    block_cls = BLOCK_REGISTRY.get(cfg.block_type)
    if block_cls is None:
        raise ValueError(
            f"Unknown block_type={cfg.block_type!r}. "
            f"Available: {sorted(BLOCK_REGISTRY.keys())}"
        )
    return block_cls(cfg)
