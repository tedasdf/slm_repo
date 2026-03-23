import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.config import AttentionConfig
from model.RoPE import RotaryEmbedding

try:
    import xformers.ops as xops

    HAS_XFORMERS = True
except ImportError:
    xops = None
    HAS_XFORMERS = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig, is_RoPE: bool):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0

        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head
        self.use_xformers = getattr(cfg, "use_xformers", False)
        self.use_qk_norm = getattr(cfg, "use_qk_norm", False)

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
            # alternative:
            # self.q_norm = nn.RMSNorm(self.head_dim)
            # self.k_norm = nn.RMSNorm(self.head_dim)

        self.is_RoPE = is_RoPE
        if self.is_RoPE:
            assert self.head_dim % 2 == 0, "head_dim must be even for RoPE."
            self.rope = RotaryEmbedding(self.head_dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(~self.mask[: q.size(-2), : k.size(-2)], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        return y

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(0, 3, 2, 1, 4)  # [B, H, 3, T, Dh]
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, H, T, Dh]

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.is_RoPE:
            cos, sin = self.rope.get_cos_sin(T, x.device, q.dtype)
            q = self.rope.apply_rotary(q, cos, sin)
            k = self.rope.apply_rotary(k, cos, sin)

        if self.use_xformers and HAS_XFORMERS and x.is_cuda:
            q_xf = q.transpose(1, 2).contiguous()  # [B, T, H, Dh]
            k_xf = k.transpose(1, 2).contiguous()
            v_xf = v.transpose(1, 2).contiguous()

            try:
                y = xops.memory_efficient_attention(
                    q_xf,
                    k_xf,
                    v_xf,
                    attn_bias=xops.LowerTriangularMask(),
                    p=self.attn_drop.p if self.training else 0.0,
                )
                y = y.transpose(1, 2).contiguous()  # [B, H, T, Dh]
            except (NotImplementedError, ValueError):
                y = self._manual_attention(q, k, v)
        else:
            y = self._manual_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
