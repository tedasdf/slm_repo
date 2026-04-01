from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


PrecisionType = Literal["fp32", "fp16", "bf16"]
DeviceType = Literal["cpu", "cuda"]


@dataclass
class TrainerConfig:
    device: DeviceType = "cuda"
    precision: PrecisionType = "bf16"

    max_steps: int = 10_000
    max_epochs: Optional[int] = None

    grad_accum_steps: int = 1
    clip_grad_norm: Optional[float] = 1.0

    train_log_every: int = 100
    eval_every: int = 1_000
    checkpoint_every: int = 1_000

    max_eval_batches: Optional[int] = None

    compile_model: bool = False
    enable_anomaly_detection: bool = False

    save_best_checkpoint: bool = True
    metric_name_for_best: str = "val_loss"
    metric_mode_for_best: Literal["min", "max"] = "min"

    num_sanity_val_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.max_epochs is not None and self.max_epochs <= 0:
            raise ValueError("max_epochs must be > 0 when provided")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be > 0")
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be > 0 when provided")
        if self.train_log_every <= 0:
            raise ValueError("train_log_every must be > 0")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be > 0")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be > 0")
        if self.max_eval_batches is not None and self.max_eval_batches <= 0:
            raise ValueError("max_eval_batches must be > 0 when provided")
        if self.num_sanity_val_steps < 0:
            raise ValueError("num_sanity_val_steps must be >= 0")