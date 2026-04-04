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

        self.attn_norm = build_norm(cfg.norm_type, cfg.model_dim)
        self.attn = build_attention(cfg)

        self.mlp_norm = build_norm(cfg.norm_type, cfg.model_dim)
        self.mlp = build_mlp(cfg)

    def forward(self, x: torch.Tensor, **attn_kwargs: Any) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), **attn_kwargs)
        x = x + self.mlp(self.mlp_norm(x))
        return x


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