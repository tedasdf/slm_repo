from __future__ import annotations

# Torch-free manifest of supported component type names.
# Derived from the ATTENTION_REGISTRY / MLP_REGISTRY / etc. in each module.
# This is what TAP (and any other tool) imports to know what's valid — no GPU
# dependencies, safe to load without a PyTorch environment.
#
# Keep in sync with:
#   attention.py  → ATTENTION_REGISTRY keys
#   mlp.py        → MLP_REGISTRY keys
#   norm.py       → build_norm accepted values
#   block.py      → BLOCK_REGISTRY keys

ATTENTION_TYPES: list[str] = [
    "baseline",   # full causal self-attention + RoPE
    "gqa",        # grouped-query attention (set attention.num_kv_heads < num_heads)
    # "swa" / "xsa" variants have key mismatches in attention.py ATTENTION_REGISTRY
    # and are not dispatchable from config yet
]

MLP_TYPES: list[str] = [
    "gelu",
    "relu2",
    "swiglu",
]

NORM_TYPES: list[str] = [
    "layernorm",
    "rmsnorm",
]

BLOCK_TYPES: list[str] = [
    "baseline",   # pre-norm: x = x + Attn(Norm(x)), x = x + MLP(Norm(x))
]

COMPONENTS: dict[str, list[str]] = {
    "attention_types": ATTENTION_TYPES,
    "mlp_types": MLP_TYPES,
    "norm_types": NORM_TYPES,
    "block_types": BLOCK_TYPES,
}
