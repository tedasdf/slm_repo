from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from slm.experiments.base import BaseExperiment, ExperimentArtifacts
from slm.model import ModelConfig, TransformerLM
from slm.training import Callback, PrintMetricsCallback, WandBCallback


@dataclass
class OptimizerConfig:
    optimizer_type: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class LoggingConfig:
    use_print_callback: bool = True
    use_wandb: bool = False
    wandb_project: str = "slm-runs"
    wandb_run_name: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=lambda: ["baseline"])


@dataclass
class BaselineExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class BaselineExperiment(BaseExperiment):
    """
    Minimal concrete experiment.

    Responsibilities:
    - builds the model
    - builds the optimizer
    - provides train/val loaders
    - optionally generates test cases
    - selects callbacks/metrics

    It does NOT own the training loop itself.
    """

    def __init__(
        self,
        cfg: BaselineExperimentConfig,
        *,
        train_loader: Any,
        val_loader: Any | None = None,
        test_loader: Any | None = None,
        extra_callbacks: list[Callback] | None = None,
        test_case_fn: Callable[[], dict[str, Any]] | None = None,
        scheduler_fn: Callable[[torch.optim.Optimizer], Any] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.extra_callbacks = extra_callbacks or []
        self.test_case_fn = test_case_fn
        self.scheduler_fn = scheduler_fn

    def build_model(self) -> TransformerLM:
        return TransformerLM(self.cfg.model)

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        opt_cfg = self.cfg.optimizer

        if opt_cfg.optimizer_type == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=opt_cfg.lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                eps=opt_cfg.eps,
                weight_decay=opt_cfg.weight_decay,
            )

        raise ValueError(
            f"Unsupported optimizer_type={opt_cfg.optimizer_type!r}. "
            f"Currently supported: ['adamw']"
        )

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> Any | None:
        if self.scheduler_fn is None:
            return None
        return self.scheduler_fn(optimizer)

    def build_dataloaders(self) -> ExperimentArtifacts:
        return ExperimentArtifacts(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
        )

    def generate_test_cases(self) -> dict[str, Any]:
        if self.test_case_fn is None:
            return {}
        return self.test_case_fn()

    def build_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = []

        log_cfg = self.cfg.logging

        if log_cfg.use_print_callback:
            callbacks.append(PrintMetricsCallback())

        if log_cfg.use_wandb:
            callbacks.append(
                WandBCallback(
                    project=log_cfg.wandb_project,
                    name=log_cfg.wandb_run_name,
                    config={
                        "model": self.cfg.model.__dict__,
                        "optimizer": self.cfg.optimizer.__dict__,
                    },
                    tags=log_cfg.wandb_tags,
                    enabled=True,
                )
            )

        callbacks.extend(self.extra_callbacks)
        return callbacks

    def build_metrics(self) -> dict[str, Any]:
        return {
            "train/loss": "Cross-entropy training loss",
            "eval/val_loss": "Validation loss",
            "diagnostics/grad_norm": "Gradient norm",
            "optimizer/lr": "Learning rate",
        }

    def build_loss_fn(self) -> Any | None:
        # Model already returns loss when targets are provided.
        return None

    def post_eval(self, trainer: Any, eval_outputs: dict[str, Any]) -> None:
        # Easy place for experiment-specific logic later.
        trainer.state.extra["experiment/last_eval_outputs"] = eval_outputs

    def post_train(self, trainer: Any) -> None:
        trainer.state.extra["experiment/finalized"] = True