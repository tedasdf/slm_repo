import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.config import AttentionConfig
from model.RoPE import RotaryEmbedding
from model.kernels.flash2_kernel import flash_attn2_triton


class FlashTritonAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig, is_RoPE: bool):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0

        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model

        self.causal = getattr(cfg, "causal", True)
        self.attn_dropout_p = getattr(cfg, "dropout", 0.0)

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.attn_drop = nn.Dropout(self.attn_dropout_p)
        self.resid_drop = nn.Dropout(self.attn_dropout_p)

        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

        self.is_RoPE = is_RoPE
        if self.is_RoPE:
            assert self.head_dim % 2 == 0, "head_dim must be even for RoPE."
            self.rope = RotaryEmbedding(self.head_dim)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def _reshape_qkv(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = (
            self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(0, 3, 1, 2, 4)
        )
        q, k, v = qkv.unbind(dim=3)  # each: [B, H, T, D]
        return q.contiguous(), k.contiguous(), v.contiguous()

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        att = torch.matmul(q, k.transpose(-1, -2)) * self.sm_scale

        if causal:
            mask = self.tril[: q.size(-2), : k.size(-2)]
            att = att.masked_fill(~mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.matmul(att, v)  # [B, H, T, D]
        return y

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        # Keep this conservative unless you know your Triton kernel supports bf16.
        return (
            x.is_cuda
            and x.dtype == torch.float16
            and self.head_dim in (16, 32, 64, 128)
        )

    def forward(self, x: torch.Tensor, causal: Optional[bool] = None) -> torch.Tensor:
        B, T, C = x.size()
        use_causal = self.causal if causal is None else causal

        q, k, v = self._reshape_qkv(x)

        if self.is_RoPE:
            cos, sin = self.rope.get_cos_sin(T, x.device, q.dtype)
            q = self.rope.apply_rotary(q, cos, sin)
            k = self.rope.apply_rotary(k, cos, sin)

        if self._can_use_triton(x) and not (
            self.training and self.attn_dropout_p > 0.0
        ):
            y = flash_attn2_triton(
                q,
                k,
                v,
                causal=use_causal,
                sm_scale=self.sm_scale,
            )
        else:
            y = self._standard_attention(q, k, v, causal=use_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y
