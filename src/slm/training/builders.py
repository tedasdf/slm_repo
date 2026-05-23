from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..data.tokenizer import BPETokenizer
from ..model import ModelConfig, TransformerLM
from .logging import PrintMetricsCallback, WandBCallback
from .trainer import Trainer

from ..data.config import DataLoaderConfig
from ..data.loaders.text_loader import build_text_dataloaders
from ..data.loaders.token_loader import build_token_dataloaders

def build_model(model_cfg: ModelConfig) -> nn.Module:
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Any | None,
) -> Any | None:
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


def build_tokenizer(tokenizer_cfg: Any | None) -> BPETokenizer | None:
    if tokenizer_cfg is None:
        return None

    tokenizer_path = getattr(tokenizer_cfg, "tokenizer_path", None)
    if tokenizer_path is None:
        return None

    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        if not getattr(tokenizer_cfg, "allow_missing_tokenizer", True):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        return None

    return BPETokenizer.load(tokenizer_path)

def build_dataloaders(
    *,
    loader_cfg: DataLoaderConfig,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
):
    mode = str(getattr(loader_cfg, "mode", "tokens")).strip().lower()

    if mode in {"text", "raw_text"}:
        return build_text_dataloaders(
            loader_cfg,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

    if mode in {"tokens", "token", "token_blocks"}:
        return build_token_dataloaders(
            loader_cfg,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

    raise ValueError(
        f"Unsupported loader mode={mode!r}. "
        "Use one of: 'text', 'raw_text', 'tokens', 'token_blocks'."
    )

def build_callbacks(
    logging_cfg: Any | None,
    *,
    enabled: bool = True,
) -> list[Any]:
    callbacks: list[Any] = []

    if logging_cfg is None or not enabled:
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
                enabled=enabled,
            )
        )

    return callbacks


def assemble_training_components(
    run_cfg: Any,
    *,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
    is_main: bool = True,
) -> dict[str, Any]:
    model = build_model(run_cfg.model)

    train_loader, val_loader = build_dataloaders(
        loader_cfg=run_cfg.data,
        rank=rank,
        world_size=world_size,
        is_distributed=is_distributed,
    )

    callbacks = build_callbacks(
        getattr(run_cfg, "logging", None),
        enabled=is_main,
    )

    tokenizer_cfg = getattr(run_cfg, "tokenizer", None)
    tokenizer = build_tokenizer(tokenizer_cfg)

    return {
        "model": model,
        "optimizer_cfg": run_cfg.optimizer,
        "scheduler_cfg": getattr(run_cfg, "scheduler", None),
        "train_loader": train_loader,
        "val_loader": val_loader,
        "callbacks": callbacks,
        "trainer_cfg": run_cfg.trainer,
        "tokenizer": tokenizer,
        "tokenizer_cfg": tokenizer_cfg,
    }


def build_trainer(run_cfg: Any, extra_callbacks: list[Any] | None = None) -> Trainer:
    parts = assemble_training_components(run_cfg)

    model = parts["model"]
    device = torch.device(
        "cuda"
        if run_cfg.trainer.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device)

    if run_cfg.trainer.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = build_optimizer(model, parts["optimizer_cfg"])
    scheduler = build_scheduler(optimizer, parts["scheduler_cfg"])

    callbacks = list(parts["callbacks"])
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=parts["train_loader"],
        val_loader=parts["val_loader"],
        config=parts["trainer_cfg"],
        callbacks=callbacks,
        tokenizer=parts.get("tokenizer"),
        tokenizer_cfg=parts.get("tokenizer_cfg"),
    )
    return trainer