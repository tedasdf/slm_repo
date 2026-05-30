from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.vocab_size = cfg.vocab_size
        self.d_model = cfg.model_dim
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def count_params(self) -> int:
        return self.embedding.weight.numel()

    def flops_per_token(self) -> float:
        return 0.0  # embedding lookup is a table index, no arithmetic