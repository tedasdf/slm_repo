from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import build_block
from .config import ModelConfig
from .embeddings import TokenEmbedding
from .norm import build_norm


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = TokenEmbedding(cfg)
        self.blocks = nn.ModuleList([build_block(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = build_norm(cfg.norm_type, cfg.model_dim)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                std = self.cfg.init.tied_embed_init_std or self.cfg.init.init_std
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.tok_emb(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if self.cfg.logit_softcap is not None:
            softcap = self.cfg.logit_softcap
            logits = softcap * torch.tanh(logits / softcap)

        out: dict[str, torch.Tensor] = {"logits": logits}

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            out["loss"] = loss

        return out