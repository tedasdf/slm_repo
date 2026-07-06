from __future__ import annotations

import math

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
        self.final_norm = build_norm(cfg.norm_type, cfg.model_dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=cfg.use_bias)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        embed_std = (
            self.cfg.init.embedding_init_std
            if self.cfg.init.embedding_init_std is not None
            else 1.0 / math.sqrt(self.cfg.model_dim)
        )
        qk_gain_init = self.cfg.attention.qk_gain_init
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=embed_std)
            elif isinstance(module, nn.Linear):
                if self.cfg.init.init_type == "fan_in":
                    # truncated-normal, std = 1/√fan_in, truncated at ±2σ
                    fan_in = module.weight.shape[1]
                    std = 1.0 / math.sqrt(fan_in)
                    if name.endswith(("q_proj", "k_proj")):
                        std *= qk_gain_init
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=std,
                                         a=-2 * std, b=2 * std)
                else:  # fixed_std
                    std = self.cfg.init.init_std
                    if name.endswith(("q_proj", "k_proj")):
                        std *= qk_gain_init
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.tok_emb(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        final_hidden = x
        logits = self.lm_head(final_hidden)

        if self.cfg.logit_softcap is not None:
            softcap = self.cfg.logit_softcap
            logits = softcap * torch.tanh(logits / softcap)

        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "final_hidden": final_hidden,
        }

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            out["loss"] = loss

        return out

    def count_params(self) -> int:
        """Total parameter count including embedding and lm_head."""
        block_params = self.cfg.num_layers * self.blocks[0].count_params()
        head_params = 0 if self.cfg.tie_embeddings else self.lm_head.weight.numel()
        return (
            self.tok_emb.count_params()
            + block_params
            + self.final_norm.count_params()
            + head_params
        )

    def count_core_params(self) -> int:
        """Parameter count excluding embedding and lm_head (Kaplan et al. convention)."""
        return (
            self.cfg.num_layers * self.blocks[0].count_params()
            + self.final_norm.count_params()
        )

    def flops_per_token(self, seq_len: int) -> float:
        block_flops = self.cfg.num_layers * self.blocks[0].flops_per_token(seq_len)
        return (
            self.tok_emb.flops_per_token()
            + block_flops
            + self.final_norm.flops_per_token()
            + self.lm_head.weight.shape[0] * self.lm_head.weight.shape[1] * 2  # lm_head matmul
        )
