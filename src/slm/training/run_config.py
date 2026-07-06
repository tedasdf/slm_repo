from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


from ..data.config import TokenizerConfig, DataLoaderConfig
from ..model import ModelConfig
from ..resource_accounting import ResourceConfig


PrecisionType = Literal["fp32", "fp16", "bf16"]
DeviceType = Literal["cpu", "cuda"]

@dataclass
class TrainerConfig:
    device: str = "cuda"
    precision: str = "bf16"

    train_tokenizer_before_fit: bool = False
    target_train_tokens: Optional[int] = None
    text_key: str = "text"
    max_seq_len: int = 1024

    max_steps: int = 10_000
    max_epochs: Optional[int] = None

    seed: int = 42

    grad_accum_steps: int = 1
    clip_grad_norm: Optional[float] = 1.0

    train_log_every: int = 100
    eval_every: int = 1_000

    save_checkpoints: bool = True
    checkpoint_every: Optional[int] = 1000
    checkpoint_dir: str = "artifacts/checkpoints"
    resume_from_checkpoint: Optional[str] = None

    max_eval_batches: Optional[int] = None

    compile_model: bool = False
    enable_anomaly_detection: bool = False

    save_best_checkpoint: bool = True
    metric_name_for_best: str = "val_loss"
    metric_mode_for_best: str = "min"

    num_sanity_val_steps: int = 0
    z_loss_coeff: float = 0.0
    independent_weight_decay: Optional[float] = None
    log_attn_logits: bool = False
    log_attention_diagnostics: bool = False
    attention_entropy_threshold: float = 0.5
    qk_spectral_iters: int = 2
    log_grad_norm_inspect: bool = False
    log_optimizer_inspect: bool = False

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
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be > 0 when provided")
        if self.max_eval_batches is not None and self.max_eval_batches <= 0:
            raise ValueError("max_eval_batches must be > 0 when provided")
        if self.num_sanity_val_steps < 0:
            raise ValueError("num_sanity_val_steps must be >= 0")
        if self.attention_entropy_threshold < 0:
            raise ValueError("attention_entropy_threshold must be >= 0")
        if self.qk_spectral_iters <= 0:
            raise ValueError("qk_spectral_iters must be > 0")


@dataclass
class OptimizerConfig:
    optimizer_type: str = "adamw"
    lr: float = 3e-4
    attention_lr_multiplier: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.attention_lr_multiplier <= 0:
            raise ValueError("attention_lr_multiplier must be > 0")

@dataclass
class SchedulerConfig:
    scheduler_type: str = "cosine_with_warmup"  # "constant" | "cosine" | "cosine_with_warmup"
    warmup_steps: int = 5_000
    t_max: int = 100_000
    eta_min: float = 1e-5

@dataclass
class LoggingConfig:
    use_print_callback: bool = True
    use_wandb: bool = False
    wandb_project: str = "slm-runs"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_config: Optional[dict] = None

@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
