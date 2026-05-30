from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResourceConfig:
    # ── GPU spec (for preflight estimation) ──────────────────────────────────
    # Peak theoretical TFLOP/s for your GPU (fp16/bf16).
    # e.g. A100-80GB=312, H100=989, RTX4090=82
    gpu_tflops: float = 312.0

    # Memory per GPU in GB.
    # e.g. A100-80GB=80, RTX4090=24
    gpu_mem_gb: float = 80.0

    # Model FLOP utilisation — realistic fraction of peak TFLOP/s.
    # 0.35–0.45 is typical for well-tuned DDP; conservative default.
    mfu: float = 0.35

    # Headroom multiplier on activation memory (accounts for framework buffers).
    activation_memory_overhead: float = 1.2

    # Fraction of GPU memory treated as usable budget.
    memory_budget_fraction: float = 0.90

    # ── Preflight behaviour ───────────────────────────────────────────────────
    run_preflight: bool = True
    abort_on_oom: bool = False

    # ── Reporting ─────────────────────────────────────────────────────────────
    log_to_wandb: bool = True
    write_json_summary: bool = True

    # How often (steps) to sample actual metrics from state.extra → W&B.
    sample_every_n_steps: int = 100

    def __post_init__(self) -> None:
        if self.gpu_tflops <= 0:
            raise ValueError("gpu_tflops must be > 0")
        if self.gpu_mem_gb <= 0:
            raise ValueError("gpu_mem_gb must be > 0")
        if not (0.0 < self.mfu <= 1.0):
            raise ValueError("mfu must be in (0, 1]")
        if self.activation_memory_overhead < 1.0:
            raise ValueError("activation_memory_overhead must be >= 1.0")
        if not (0.0 < self.memory_budget_fraction <= 1.0):
            raise ValueError("memory_budget_fraction must be in (0, 1]")
        if self.sample_every_n_steps <= 0:
            raise ValueError("sample_every_n_steps must be > 0")