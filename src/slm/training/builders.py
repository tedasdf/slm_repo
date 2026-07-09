from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.slm.data.tokenizer import AnyTokenizer, BPETokenizer, SentencePieceTokenizer
from src.slm.model import ModelConfig, TransformerLM
from .logging import AttnLogitCallback, PrintMetricsCallback, WandBCallback
from .trainer import Trainer
from ..utils.seed import seed_everything

from src.slm.data.config import DataLoaderConfig
from src.slm.data.loaders.text_loader import build_packed_text_dataloaders, build_text_dataloaders
from src.slm.data.loaders.token_loader import build_token_dataloaders

def build_model(model_cfg: ModelConfig, precision: str = "bf16") -> nn.Module:
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    half_precision = precision in ("fp16", "bf16")
    enable_cudnn_sdp(False)
    enable_flash_sdp(half_precision)        # flash requires fp16/bf16
    enable_mem_efficient_sdp(False)
    enable_math_sdp(not half_precision)     # math kernel is the fp32 fallback

    return TransformerLM(model_cfg)


def build_optimizer(model: nn.Module, optimizer_cfg: Any) -> torch.optim.Optimizer:
    optimizer_type = getattr(optimizer_cfg, "optimizer_type", "adamw").lower()
    lr = optimizer_cfg.lr
    attention_lr_multiplier = float(getattr(optimizer_cfg, "attention_lr_multiplier", 1.0))

    params: Any
    if attention_lr_multiplier != 1.0:
        attn_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ".attn." in name:
                attn_params.append(param)
            else:
                other_params.append(param)

        params = []
        if other_params:
            params.append({"params": other_params, "lr": lr, "name": "default"})
        if attn_params:
            params.append(
                {
                    "params": attn_params,
                    "lr": lr * attention_lr_multiplier,
                    "name": "attention",
                }
            )
    else:
        params = model.parameters()

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
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
    eta_min = getattr(scheduler_cfg, "eta_min", 1e-5)

    if scheduler_type == "cosine_with_warmup":
        warmup_steps = int(getattr(scheduler_cfg, "warmup_steps", 5_000))
        t_max = int(getattr(scheduler_cfg, "t_max", 100_000))

        if warmup_steps >= t_max:
            raise ValueError(
                f"warmup_steps={warmup_steps} must be < t_max={t_max}"
            )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / max(warmup_steps, 1),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max - warmup_steps,
            eta_min=eta_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    if scheduler_type == "cosine":
        t_max = int(getattr(scheduler_cfg, "t_max", 100_000))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    if scheduler_type == "constant":
        return None

    raise ValueError(
        f"Unsupported scheduler_type={scheduler_type!r}. "
        f"Supported: ['cosine_with_warmup', 'cosine', 'constant']"
    )


def build_tokenizer(tokenizer_cfg: Any | None) -> AnyTokenizer | None:
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

    tokenizer_type = getattr(tokenizer_cfg, "tokenizer_type", "bpe").lower()
    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer.load(tokenizer_path)
    return BPETokenizer.load(tokenizer_path)

def build_dataloaders(
    *,
    loader_cfg: DataLoaderConfig,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
    tokenizer: Any | None = None,
    tokenizer_cfg: Any | None = None,
):
    mode = str(getattr(loader_cfg, "mode", "tokens")).strip().lower()

    if mode == "packed_text":
        if tokenizer is None:
            raise ValueError(
                "mode='packed_text' requires a pre-loaded tokenizer. "
                "Set tokenizer_path and allow_missing_tokenizer: false in the tokenizer config."
            )
        return build_packed_text_dataloaders(
            loader_cfg,
            tokenizer,
            tokenizer_cfg,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

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
        "Use one of: 'packed_text', 'text', 'raw_text', 'tokens', 'token_blocks'."
    )

def build_callbacks(
    logging_cfg: Any | None,
    *,
    enabled: bool = True,
    yaml_path: str | None = None,
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
                entity=getattr(logging_cfg, "wandb_entity", None),
                run_id=getattr(logging_cfg, "wandb_run_id", None),
                config=getattr(logging_cfg, "wandb_config", None),
                tags=getattr(logging_cfg, "wandb_tags", None),
                enabled=enabled,
                yaml_path=yaml_path,
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
    yaml_path: str | None = None,
) -> dict[str, Any]:
    seed = getattr(run_cfg.trainer, "seed", 42)
    seed_everything(seed, rank=rank)

    model = build_model(run_cfg.model, precision=getattr(run_cfg.trainer, "precision", "bf16"))

    tokenizer_cfg = getattr(run_cfg, "tokenizer", None)
    tokenizer = build_tokenizer(tokenizer_cfg)

    train_loader, val_loader = build_dataloaders(
        loader_cfg=run_cfg.data,
        rank=rank,
        world_size=world_size,
        is_distributed=is_distributed,
        tokenizer=tokenizer,
        tokenizer_cfg=tokenizer_cfg,
    )

    callbacks = build_callbacks(
        getattr(run_cfg, "logging", None),
        enabled=is_main,
        yaml_path=yaml_path,
    )

    # Imported lazily: resource_accounting -> training.callbacks -> (full
    # training package init) -> training.builders -> resource_accounting
    # would deadlock on a module-level import if resource_accounting is
    # imported before training.
    # from src.slm.resource_accounting import ResourceAccountingCallback

    # callbacks.append(
    #     ResourceAccountingCallback(
    #         model_cfg=run_cfg.model,
    #         trainer_cfg=run_cfg.trainer,
    #         resource_cfg=getattr(run_cfg, "resource", None),
    #         batch_size=run_cfg.data.batch_size,
    #     )
    # )

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
