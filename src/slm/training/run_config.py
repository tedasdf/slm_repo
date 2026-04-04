from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from src.slm.model import ModelConfig


PrecisionType = Literal["fp32", "fp16", "bf16"]
DeviceType = Literal["cpu", "cuda"]


@dataclass
class TrainerConfig:
    device: str = "cuda"
    precision: str = "bf16"

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
    metric_mode_for_best: str = "min"

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

@dataclass
class DataLoaderConfig:
    train_bin_path: str = "artifacts/tokenizer/latest/splits/train.bin"
    val_bin_path: Optional[str] = "artifacts/tokenizer/latest/splits/val.bin"

    seq_len: int = 1024
    batch_size: int = 8

    stride: Optional[int] = None   # None -> use seq_len
    shuffle_train: bool = True

    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True

@dataclass
class OptimizerConfig:
    optimizer_type: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    scheduler_type: str = "constant"   # "constant" | "cosine"
    t_max: Optional[int] = None
    eta_min: float = 0.0


@dataclass
class LoggingConfig:
    use_print_callback: bool = True
    use_wandb: bool = False
    wandb_project: str = "slm-runs"
    wandb_run_name: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_config: Optional[dict] = None




@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)