from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)