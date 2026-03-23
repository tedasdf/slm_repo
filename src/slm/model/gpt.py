import torch.nn as nn
import torch

from torch.nn import functional as F
from model.attention import get_attention
from model.config import AttentionConfig, MLPConfig, GPTConfig
from mlp import StandardMLP, SwiGLUMLP


def build_mlp(cfg):
    if cfg.mlp_type == "standard":
        return StandardMLP(cfg)
    elif cfg.mlp_type == "swiglu":
        return SwiGLUMLP(cfg)
    else:
        raise ValueError(f"Unknown mlp_type: {cfg.mlp_type}")


class Block(nn.Module):
    def __init__(self, attn_cfg: AttentionConfig, mlp_cfg: MLPConfig, is_RoPE: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(attn_cfg.d_model)
        self.attn = get_attention(attn_cfg.attn_type, cfg=attn_cfg, is_RoPE=is_RoPE)
        self.ln2 = nn.LayerNorm(mlp_cfg.d_model)
        self.mlp = build_mlp(mlp_cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.is_RoPE = cfg.is_RoPE
        if not self.is_RoPE:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg.attn, cfg.mlp, self.is_RoPE) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._apply_depth_scaled_init()
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_depth_scaled_init(self):
        scale = (2 * self.cfg.n_layer) ** -0.5

        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            is_residual_output = (
                name.endswith("attn.proj")
                or name.endswith("mlp.net.2")
                or name.endswith("mlp.w_out")
            )

            if is_residual_output:
                nn.init.normal_(module.weight, mean=0.0, std=0.02 * scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)

        if self.is_RoPE:
            x = self.drop(tok)
        else:
            pos = self.pos_emb[:, :T, :]
            x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean"
            )

        return logits, loss
