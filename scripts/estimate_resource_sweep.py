"""
Preflight resource estimates for a depth x width architecture sweep.

Prints param count, memory breakdown, per-step time, and total wall-clock
time for each (num_layers, model_dim) combination, using the same
ResourceEstimator formulas the training run uses for its preflight check.

Run from the repo root:
    python scripts/estimate_resource_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.slm.model.config import AttentionConfig, MLPConfig, ModelConfig
from src.slm.resource_accounting import ResourceConfig, check_budget, estimate_resources
from src.slm.training.run_config import TrainerConfig

# ── sweep grid ──────────────────────────────────────────────────────────────

DEPTHS = 24 # [3, 6, 12, 24]
WIDTHS = [2048] # [128, 256, 768, 1024, 2048]
NUM_HEADS = 8
NUM_KV_HEADS = 8

BATCH_SIZE = 256
GRAD_ACCUM_STEPS = 1
MAX_SEQ_LEN = 512
MAX_STEPS = 100_000

# L40S: 48GB, ~181 TFLOPs bf16 (dense, non-sparse)
GPU_TFLOPS = 181.0
GPU_MEM_GB = 48.0
MFU = 0.35


def make_trainer_cfg() -> TrainerConfig:
    return TrainerConfig(
        device="cuda",
        precision="bf16",
        max_seq_len=MAX_SEQ_LEN,
        max_steps=MAX_STEPS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
    )


def make_resource_cfg() -> ResourceConfig:
    return ResourceConfig(gpu_tflops=GPU_TFLOPS, gpu_mem_gb=GPU_MEM_GB, mfu=MFU)


def make_model_cfg(depth: int, width: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=32101,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=depth,
        model_dim=width,
        norm_type="layernorm",
        norm_eps=1e-6,
        use_bias=False,
        tie_embeddings=False,
        attention=AttentionConfig(num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, qk_norm=False),
        mlp=MLPConfig(mlp_type="gelu", mlp_mult=4.0),
    )


def main() -> None:
    trainer_cfg = make_trainer_cfg()
    resource_cfg = make_resource_cfg()

    header = (
        f"{'depth':>5} {'width':>5} {'params(M)':>10} {'param':>7} {'act':>7} "
        f"{'opt':>7} {'total_gb':>9} {'step_s':>8} {'hrs@100k':>9}  budget"
    )
    print(header)

    for depth in DEPTHS:
        for width in WIDTHS:
            model_cfg = make_model_cfg(depth, width)
            est = estimate_resources(
                model_cfg, trainer_cfg, resource_cfg, batch_size=BATCH_SIZE, world_size=1
            )
            budget = check_budget(est, resource_cfg, world_size=1)
            print(
                f"{depth:>5} {width:>5} {est.num_params / 1e6:>10.2f} "
                f"{est.param_mem_gb:>7.3f} {est.activation_mem_gb:>7.3f} "
                f"{est.optimizer_mem_gb:>7.3f} {est.total_mem_gb:>9.3f} "
                f"{est.est_step_time_sec:>8.4f} {est.est_total_hours:>9.2f}  {budget.status.value}"
            )


if __name__ == "__main__":
    main()
