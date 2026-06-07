from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .norm import RMSNorm
from .rope import apply_rope, build_rope_cache




class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.model_dim = cfg.model_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.num_kv_heads = cfg.num_kv_heads

        bias = cfg.use_bias
        self.q_proj = nn.Linear(self.model_dim, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.model_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.model_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.model_dim, bias=bias)

        self.is_gqa = self.num_kv_heads != self.num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.q_per_kv = self.num_heads // self.num_kv_heads

        # QK-norm: one RMSNorm of size head_dim, shared across all heads
        if cfg.attention.qk_norm:
            self.q_norm: RMSNorm | None = RMSNorm(self.head_dim, eps=cfg.norm_eps)
            self.k_norm: RMSNorm | None = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def _reshape_q(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        return q.transpose(1, 2)  # [B, H, T, D]

    def _reshape_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape

        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        if self.k_norm is not None:
            k = self.k_norm(k)
        k = k.transpose(1, 2)

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

    def count_params(self) -> int:
        projs = [self.q_proj, self.k_proj, self.v_proj, self.out_proj]
        total = sum(p.weight.numel() for p in projs)
        total += sum(p.bias.numel() for p in projs if p.bias is not None)
        if self.q_norm is not None:
            total += self.q_norm.count_params()
        if self.k_norm is not None:
            total += self.k_norm.count_params()
        return total

    def flops_per_token(self, seq_len: int) -> float:
        q_term      = 2 * self.model_dim * self.num_heads * self.head_dim
        k_term      = 2 * self.model_dim * self.num_kv_heads * self.head_dim
        v_term      = 2 * self.model_dim * self.num_kv_heads * self.head_dim
        out_term    = 2 * self.num_heads * self.head_dim * self.model_dim

        qk          = 2 * self.num_heads * self.head_dim * seq_len
        softmax     = seq_len + (seq_len - 1) + seq_len  # per head
        softmax    *= self.num_heads
        scores_v    = 2 * self.num_heads * seq_len * self.head_dim

        # QK-norm: one RMSNorm of head_dim applied per head per token
        qk_norm_flops = 0.0
        if self.q_norm is not None:
            qk_norm_flops += self.num_heads * self.q_norm.flops_per_token()
        if self.k_norm is not None:
            qk_norm_flops += self.num_kv_heads * self.k_norm.flops_per_token()

        return q_term + k_term + v_term + out_term + qk + softmax + scores_v + qk_norm_flops


class ExclusionSelfAttention(CausalSelfAttention):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)

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

    def flops_per_token(self, seq_len: int) -> float:
        q_term      = 2 * self.model_dim * self.num_heads * self.head_dim
        k_term      = 2 * self.model_dim * self.num_kv_heads * self.head_dim
        v_term      = 2 * self.model_dim * self.num_kv_heads * self.head_dim
        out_term    = 2 * self.num_heads * self.head_dim * self.model_dim

        # XSA uses full causal attention, not a sliding window
        qk          = 2 * self.num_heads * self.head_dim * seq_len
        softmax     = seq_len + (seq_len - 1) + seq_len
        softmax    *= self.num_heads
        scores_v    = 2 * self.num_heads * seq_len * self.head_dim

        norm_term = self.head_dim + (self.head_dim - 1) + 1 + self.head_dim  # squares + sum + sqrt + divide
        excl_term = self.head_dim + self.head_dim - 1 + self.head_dim + self.head_dim
        norm_excel = (norm_term + excl_term) * self.num_heads
        return q_term + k_term + v_term + out_term + qk + softmax + scores_v + norm_excel



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

    def flops_per_token(self, seq_len: int) -> float:
        q_term      = 2 * self.model_dim * self.num_heads * self.head_dim
        k_term      = 2 * self.model_dim * self.num_kv_heads * self.head_dim
        v_term      = 2 * self.model_dim * self.num_kv_heads * self.head_dim
        out_term    = 2 * self.num_heads * self.head_dim * self.model_dim

        qk          = 2 * self.num_heads * self.head_dim * self.window_size
        softmax     = self.window_size + (self.window_size - 1) + self.window_size
        softmax    *= self.num_heads
        scores_v    = 2 * self.num_heads * self.window_size * self.head_dim

        return q_term + k_term + v_term + out_term + qk + softmax + scores_v




class ResidualAttention():
    def __init__():
        raise ValueError


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