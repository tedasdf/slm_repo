from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn



from src.slm.training.lm_loader import build_dataloaders as build_token_dataloaders
from slm.model import ModelConfig, TransformerLM
from slm.training.logging import PrintMetricsCallback, WandBCallback
from slm.training.trainer import Trainer
from slm.training.run_config import TrainerConfig



def build_model(model_cfg: ModelConfig) -> nn.Module:
    return TransformerLM(model_cfg)


def build_optimizer(model: nn.Module, optimizer_cfg: Any) -> torch.optim.Optimizer:
    optimizer_type = getattr(optimizer_cfg, "optimizer_type", "adamw").lower()

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_cfg.lr,
            betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
            eps=optimizer_cfg.eps,
            weight_decay=optimizer_cfg.weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer_type={optimizer_type!r}. "
        f"Currently supported: ['adamw']"
    )


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_cfg: Any | None) -> Any | None:
    if scheduler_cfg is None:
        return None

    scheduler_type = getattr(scheduler_cfg, "scheduler_type", None)
    if scheduler_type is None:
        return None

    scheduler_type = scheduler_type.lower()

    if scheduler_type == "cosine":
        t_max = getattr(scheduler_cfg, "t_max", None)
        eta_min = getattr(scheduler_cfg, "eta_min", 0.0)

        if t_max is None:
            raise ValueError("scheduler_type='cosine' requires scheduler_cfg.t_max")

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    if scheduler_type == "constant":
        return None

    raise ValueError(
        f"Unsupported scheduler_type={scheduler_type!r}. "
        f"Currently supported: ['cosine', 'constant']"
    )


def build_dataloaders(data_cfg: Any):
    return build_token_dataloaders(data_cfg)


def build_callbacks(logging_cfg: Any | None) -> list[Any]:
    callbacks: list[Any] = []

    if logging_cfg is None:
        return callbacks

    if getattr(logging_cfg, "use_print_callback", False):
        callbacks.append(PrintMetricsCallback())

    if getattr(logging_cfg, "use_wandb", False):
        callbacks.append(
            WandBCallback(
                project=getattr(logging_cfg, "wandb_project", "slm-runs"),
                name=getattr(logging_cfg, "wandb_run_name", None),
                config=getattr(logging_cfg, "wandb_config", None),
                tags=getattr(logging_cfg, "wandb_tags", None),
                enabled=True,
            )
        )

    return callbacks

def build_trainer(run_cfg: Any, extra_callbacks: list[Any] | None = None) -> Trainer:
    parts = assemble_training_components(run_cfg)

    callbacks = list(parts["callbacks"])
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    trainer = Trainer(
        model=parts["model"],
        optimizer=parts["optimizer"],
        scheduler=parts["scheduler"],
        train_loader=parts["train_loader"],
        val_loader=parts["val_loader"],
        config=parts["trainer_cfg"],
        callbacks=callbacks,
    )
    return trainer



@dataclass
class TrainingComponents:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any | None
    train_loader: Any
    val_loader: Any | None
    callbacks: list[Any]
    trainer_cfg: TrainerConfig

