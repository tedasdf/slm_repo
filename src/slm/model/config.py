from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


AttentionType = Literal[
    "baseline",
    "gqa",
    "swa",
    "gqa_swa",
    "xsa",
    "xsa_gqa",
    "xsa_swa",
    "xsa_gqa_swa",
]

MLPType = Literal["gelu", "swiglu", "relu2"]
NormType = Literal["layernorm", "rmsnorm"]
BlockType = Literal["baseline"]


@dataclass
class AttentionConfig:
    attention_type: str = "baseline"

    # attention shape
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    # positional / attention options
    rope_base: float = 10_000.0
    qk_gain_init: float = 1.0
    window_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")

        if self.num_kv_heads is not None:
            if self.num_kv_heads <= 0:
                raise ValueError("num_kv_heads must be > 0")
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError("num_heads must be divisible by num_kv_heads")

        if "swa" in self.attention_type and self.window_size is None:
            raise ValueError("window_size must be set when using an swa attention type")

        if self.window_size is not None and self.window_size <= 0:
            raise ValueError("window_size must be > 0")

        if self.head_dim is not None and self.head_dim <= 0:
            raise ValueError("head_dim must be > 0")

        if self.rope_base <= 0:
            raise ValueError("rope_base must be > 0")


@dataclass
class MLPConfig:
    mlp_type: str = "swiglu"

    # choose one of:
    # - hidden_dim directly
    # - mlp_mult * model_dim
    hidden_dim: Optional[int] = None
    mlp_mult: float = 4.0

    def __post_init__(self) -> None:
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.mlp_mult <= 0:
            raise ValueError("mlp_mult must be > 0")


@dataclass
class InitConfig:
    init_std: float = 0.02
    tied_embed_init_std: Optional[float] = None

    def __post_init__(self) -> None:
        if self.init_std <= 0:
            raise ValueError("init_std must be > 0")
        if self.tied_embed_init_std is not None and self.tied_embed_init_std <= 0:
            raise ValueError("tied_embed_init_std must be > 0")


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    max_seq_len: int = 1024

    num_layers: int = 8
    model_dim: int = 512

    block_type: str = "baseline"
    norm_type: str = "rmsnorm"
    tie_embeddings: bool = True
    logit_softcap: Optional[float] = None

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    init: InitConfig = field(default_factory=InitConfig)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.model_dim <= 0:
            raise ValueError("model_dim must be > 0")

        if self.logit_softcap is not None and self.logit_softcap <= 0:
            raise ValueError("logit_softcap must be > 0 when provided")

        # derive / validate attention dimensions
        if self.attention.head_dim is None:
            if self.model_dim % self.attention.num_heads != 0:
                raise ValueError(
                    "model_dim must be divisible by attention.num_heads when head_dim is not set"
                )
            self.attention.head_dim = self.model_dim // self.attention.num_heads
        else:
            expected_model_dim = self.attention.num_heads * self.attention.head_dim
            if expected_model_dim != self.model_dim:
                raise ValueError(
                    "model_dim must equal attention.num_heads * attention.head_dim "
                    "when head_dim is provided explicitly"
                )

        if self.attention.head_dim % 2 != 0:
            raise ValueError("attention.head_dim must be even for RoPE")

        if self.attention.num_kv_heads is None:
            if "gqa" in self.attention.attention_type:
                raise ValueError(
                    "attention.num_kv_heads must be set for gqa attention types"
                )
            self.attention.num_kv_heads = self.attention.num_heads

        # derive / validate MLP hidden size
        if self.mlp.hidden_dim is None:
            self.mlp.hidden_dim = int(self.mlp.mlp_mult * self.model_dim)

        if self.mlp.hidden_dim <= 0:
            raise ValueError("mlp.hidden_dim must be > 0")

    @property
    def head_dim(self) -> int:
        return self.attention.head_dim  # type: ignore[return-value]

    @property
    def num_heads(self) -> int:
        return self.attention.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self.attention.num_kv_heads  # type: ignore[return-value]

    @property
    def hidden_dim(self) -> int:
        return self.mlp.hidden_dim  # type: ignore[return-value]