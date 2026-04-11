from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from src.slm.data.config import DatasetConfig
from src.slm.model import ModelConfig


PrecisionType = Literal["fp32", "fp16", "bf16"]
DeviceType = Literal["cpu", "cuda"]


@dataclass
class TrainerConfig:
    device: str = "cuda"
    precision: str = "bf16"

    world_size: int = 1
    
    train_tokenizer_before_fit: bool = False
    text_key: str = "text"
    max_seq_len: int = 1024

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
    # --------------------------------------------------
    # mode switch
    # --------------------------------------------------
    use_online_tokenization: bool = False

    # --------------------------------------------------
    # pretokenized / bin-loader mode
    # --------------------------------------------------
    train_bin_path: Optional[str] = "artifacts/tokenizer/latest/splits/train.bin"
    val_bin_path: Optional[str] = "artifacts/tokenizer/latest/splits/val.bin"

    seq_len: int = 1024
    stride: Optional[int] = None   # None -> use seq_len

    # --------------------------------------------------
    # raw-text / online-tokenization mode
    # --------------------------------------------------
    source_type: str = "huggingface"
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None

    train_split_name: str = "train"
    val_split_name: Optional[str] = None
    test_split_name: Optional[str] = None

    text_fields: list[str] = field(default_factory=lambda: ["text"])

    cache_dir: Optional[str] = None
    streaming: bool = False

    seed: int = 42
    shuffle: bool = True
    shuffle_buffer_size: int = 10_000

    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    smoke_test: bool = False

    # for local json.gz / dolma-style sources
    data_files_glob: Optional[str] = None

    # --------------------------------------------------
    # dataloader behavior
    # --------------------------------------------------
    batch_size: int = 8
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
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    logging: LoggingConfig
    trainer: TrainerConfig
    data: DataLoaderConfig
    dataset: DatasetConfig
    tokenizer: TokenizerConfig