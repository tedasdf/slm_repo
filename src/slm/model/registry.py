from __future__ import annotations

import torch.nn as nn

from .attention import ATTENTION_REGISTRY
from .block import BLOCK_REGISTRY
from .mlp import MLP_REGISTRY
from .norm import RMSNorm

NORM_REGISTRY: dict[str, type[nn.Module]] = {
    "rmsnorm": RMSNorm,
    "layernorm": nn.LayerNorm,
}


def get_attention_cls(name: str) -> type[nn.Module]:
    if name not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention type '{name}'. Available: {sorted(ATTENTION_REGISTRY)}"
        )
    return ATTENTION_REGISTRY[name]


def get_mlp_cls(name: str) -> type[nn.Module]:
    if name not in MLP_REGISTRY:
        raise ValueError(
            f"Unknown mlp type '{name}'. Available: {sorted(MLP_REGISTRY)}"
        )
    return MLP_REGISTRY[name]


def get_norm_cls(name: str) -> type[nn.Module]:
    if name not in NORM_REGISTRY:
        raise ValueError(
            f"Unknown norm type '{name}'. Available: {sorted(NORM_REGISTRY)}"
        )
    return NORM_REGISTRY[name]


def get_block_cls(name: str) -> type[nn.Module]:
    if name not in BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown block type '{name}'. Available: {sorted(BLOCK_REGISTRY)}"
        )
    return BLOCK_REGISTRY[name]
