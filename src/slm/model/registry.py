from __future__ import annotations

import torch.nn as nn

from .attention import ATTENTION_REGISTRY
from .block import BLOCK_REGISTRY
from .mlp import MLP_REGISTRY
from .norm import RMSNorm

NORM_REGISTRY: dict[str, type[nn.Module]] = {
    "rmsnorm": RMSNorm,
    "baseline": RMSNorm,
}


def _get(registry: dict[str, type[nn.Module]], name: str, kind: str) -> type[nn.Module]:
    if name not in registry:
        raise ValueError(f"Unknown {kind} {name!r}. Available: {sorted(registry)}")
    return registry[name]


def get_attention_cls(name: str) -> type[nn.Module]:
    return _get(ATTENTION_REGISTRY, name, "attention_type")


def get_mlp_cls(name: str) -> type[nn.Module]:
    return _get(MLP_REGISTRY, name, "mlp_type")


def get_norm_cls(name: str) -> type[nn.Module]:
    return _get(NORM_REGISTRY, name, "norm_type")


def get_block_cls(name: str) -> type[nn.Module]:
    return _get(BLOCK_REGISTRY, name, "block_type")
