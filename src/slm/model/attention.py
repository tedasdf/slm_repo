from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import apply_rope, build_rope_cache




class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.model_dim = cfg.model_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.num_kv_heads = cfg.num_kv_heads

        self.q_proj = nn.Linear(self.model_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.model_dim, bias=False)

        self.is_gqa = self.num_kv_heads != self.num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.q_per_kv = self.num_heads // self.num_kv_heads

    def _reshape_q(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        return q.transpose(1, 2)  # [B, H, T, D]

    def _reshape_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape

        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.is_gqa:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)

        return k, v

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self._reshape_q(x)
        k, v = self._reshape_kv(x)

        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.head_dim,
            base=self.cfg.attention.rope_base,
            device=x.device,
            dtype=x.dtype,
        )
        q, k = apply_rope(q, k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(y)

class ExclusionSelfAttention(CausalSelfAttention):
    def __init__(self, cfg: ModelConfig) -> None:
        super.__init__(cfg)
    
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self._reshape_q(x)          # (B, Hq, L, D)
        k, v = self._reshape_kv(x)      # (B, Hkv, L, D)

        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.head_dim,
            base=self.cfg.attention.rope_base,
            device=x.device,
            dtype=x.dtype,
        )
        q, k = apply_rope(q, k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=(self.num_heads != self.num_kv_heads),
        )

        if self.num_heads != self.num_kv_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            v_excl = v.repeat_interleave(repeat_factor, dim=1)
        else:
            v_excl = v

        vn = F.normalize(v_excl, dim=-1)
        z = y - (y * vn).sum(dim=-1, keepdim=True) * vn

        z = z.transpose(1, 2).contiguous().view(
            bsz, seq_len, self.num_heads * self.head_dim
        )
        return self.out_proj(z)

class SlidingWindowAttention(CausalSelfAttention):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.window_size = cfg.attention.window_size

    def _build_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        q_pos = idx[:, None]
        k_pos = idx[None, :]

        return (k_pos <= q_pos) & (k_pos >= q_pos - self.window_size + 1)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self._reshape_q(x)
        k, v = self._reshape_kv(x)

        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.head_dim,
            base=self.cfg.attention.rope_base,
            device=x.device,
            dtype=x.dtype,
        )
        q, k = apply_rope(q, k, cos, sin)

        mask = self._build_sliding_window_mask(seq_len, x.device)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=(self.num_heads != self.num_kv_heads),
        )

        y = y.transpose(1, 2).contiguous().view(
            bsz, seq_len, self.num_heads * self.head_dim
        )
        return self.out_proj(y)

ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline": CausalSelfAttention,
    "gqa": CausalSelfAttention,
    'SlidingWindow': SlidingWindowAttention,
    'XSA': ExclusionSelfAttention,
}


def build_attention(cfg: ModelConfig) -> nn.Module:
    attn_cls = ATTENTION_REGISTRY.get(cfg.attention.attention_type)
    if attn_cls is None:
        raise ValueError(
            f"Unsupported attention_type={cfg.attention.attention_type!r} in the simple model stack. "
            f"Supported: {sorted(ATTENTION_REGISTRY.keys())}"
        )
    return attn_cls(cfg)