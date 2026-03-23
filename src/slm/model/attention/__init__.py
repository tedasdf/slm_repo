# model/attention/__init__.py
from .standard import CausalSelfAttention
from .flash_attn import FlashTritonAttention

ATTENTION_REGISTRY = {
    "standard": CausalSelfAttention,
    "flash": FlashTritonAttention
}


def get_attention(name, **kwargs):
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention type: {name}")
    return ATTENTION_REGISTRY[name](**kwargs)
