from dataclasses import dataclass

@dataclass
class BaseAttentionConfig:
    attn_type: str
    d_model: int
    n_head: int
    block_size: int
    dropout: float


@dataclass
class StandardAttentionConfig(BaseAttentionConfig):
    pass


@dataclass
class MLPConfig:
    mlp_type: str
    d_model: int
    dropout: float


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    d_model: int
    block_size: int
    dropout: int
    is_RoPE: bool
    # Nested configs
    attn: BaseAttentionConfig
    mlp: MLPConfig

   