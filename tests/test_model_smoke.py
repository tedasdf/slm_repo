import torch

from src.slm.model import ModelConfig, TransformerLM
from src.slm.model.attention import ATTENTION_REGISTRY
from src.slm.model.config import AttentionConfig, InitConfig, MLPConfig


def make_tiny_config(attention_type: str = "baseline") -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        max_seq_len=16,
        num_layers=2,
        model_dim=64,
        block_type="baseline",
        norm_type="rmsnorm",
        attention=AttentionConfig(
            attention_type=attention_type,
            num_heads=4,
            num_kv_heads=4,
            head_dim=None,
            window_size=8 if "swa" in attention_type else None,
        ),
        mlp=MLPConfig(mlp_type="swiglu", mlp_mult=2.0),
        init=InitConfig(init_std=0.02),
    )


def test_attention_registry_uses_canonical_names():
    assert "baseline" in ATTENTION_REGISTRY
    assert "gqa" in ATTENTION_REGISTRY
    assert "swa" in ATTENTION_REGISTRY
    assert "xsa" in ATTENTION_REGISTRY
    assert "SlidingWindow" not in ATTENTION_REGISTRY
    assert "XSA" not in ATTENTION_REGISTRY


def test_transformer_forward_smoke():
    torch.manual_seed(0)
    cfg = make_tiny_config()
    model = TransformerLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids, targets=input_ids)
    assert out["logits"].shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
