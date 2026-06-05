"""
Verify that two runs with the same seed produce identical results.

Checks:
  1. Model weights are identical after init.
  2. First training step produces the same loss.
  3. Model weights are identical after one update step.

Run from the repo root:
    python scripts/verify_seed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from slm.model.config import AttentionConfig, MLPConfig, ModelConfig
from slm.model.model import TransformerLM
from slm.utils.seed import seed_everything


# ── tiny model config ─────────────────────────────────────────────────────────

def make_model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=32,
        num_layers=2,
        model_dim=64,
        attention=AttentionConfig(num_heads=4),
        mlp=MLPConfig(mlp_type="swiglu", mlp_mult=2.0),
    )


def make_batch(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    ids = torch.randint(0, 256, (2, 32), generator=g)
    return {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}


def build_and_run(seed: int) -> dict:
    seed_everything(seed)
    cfg = make_model_cfg()
    model = TransformerLM(cfg)

    weights_before = {
        k: v.clone() for k, v in model.named_parameters()
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_batch(seed=0)  # same data every time — only RNG state differs

    model.train()
    optimizer.zero_grad()
    out = model(**batch)
    loss = out["loss"]
    loss.backward()
    optimizer.step()

    weights_after = {
        k: v.clone() for k, v in model.named_parameters()
    }

    return {
        "loss": loss.item(),
        "weights_before": weights_before,
        "weights_after": weights_after,
    }


# ── checks ────────────────────────────────────────────────────────────────────

def assert_dicts_equal(a: dict, b: dict, label: str) -> None:
    mismatches = []
    for k in a:
        if not torch.equal(a[k], b[k]):
            max_diff = (a[k] - b[k]).abs().max().item()
            mismatches.append(f"  {k}: max_diff={max_diff:.2e}")
    if mismatches:
        print(f"FAIL — {label} mismatch:")
        for m in mismatches:
            print(m)
        sys.exit(1)
    print(f"  OK  {label}")


def main() -> None:
    SEED = 42
    print(f"Running two identical runs with seed={SEED} …\n")

    run_a = build_and_run(SEED)
    run_b = build_and_run(SEED)

    print("Checking …")
    assert_dicts_equal(run_a["weights_before"], run_b["weights_before"], "weights after init")

    if run_a["loss"] != run_b["loss"]:
        print(f"FAIL — loss mismatch: {run_a['loss']} vs {run_b['loss']}")
        sys.exit(1)
    print(f"  OK  loss ({run_a['loss']:.6f})")

    assert_dicts_equal(run_a["weights_after"], run_b["weights_after"], "weights after one step")

    print("\nAll checks passed — seed control is working correctly.")


if __name__ == "__main__":
    main()
