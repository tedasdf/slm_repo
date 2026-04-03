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



class XSACausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        window_size: int | None = None,  # ignored, kept for constructor compatibility
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_kv_heads != num_heads:
            raise ValueError("XSACausalSelfAttention expects num_kv_heads == num_heads")
        if window_size is not None:
            raise ValueError("XSACausalSelfAttention does not use window_size")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.model_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = build_rope_cache(
            seq_len=seqlen,
            head_dim=self.head_dim,
            base=self.cfg.attention.rope_base,
            device=x.device,
            dtype=x.dtype,
        )
        q, k = apply_rope(q, k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=False,
        )
        z = _apply_xsa_projection(y, v)

        z = z.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(z)



ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline": CausalSelfAttention,
    "gqa": CausalSelfAttention,
}


def build_attention(cfg: ModelConfig) -> nn.Module:
    attn_cls = ATTENTION_REGISTRY.get(cfg.attention.attention_type)
    if attn_cls is None:
        raise ValueError(
            f"Unsupported attention_type={cfg.attention.attention_type!r} in the simple model stack. "
            f"Supported: {sorted(ATTENTION_REGISTRY.keys())}"
        )
    return attn_cls(cfg)