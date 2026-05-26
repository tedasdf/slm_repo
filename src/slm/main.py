from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from .training.builders import (
    assemble_training_components,
    build_optimizer,
    build_scheduler,
)
from .training.distributed import cleanup_distributed, setup_distributed
from .training.run_config import RunConfig
from .training.trainer import Trainer


def load_config(config_path: str | Path) -> RunConfig:
    schema = OmegaConf.structured(RunConfig)
    loaded_cfg = OmegaConf.load(str(config_path))
    merged = OmegaConf.merge(schema, loaded_cfg)

    missing = OmegaConf.missing_keys(merged)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")

    cfg = OmegaConf.to_object(merged)
    if not isinstance(cfg, RunConfig):
        raise TypeError(f"Expected RunConfig, got {type(cfg)}")
    return cfg


def main(config_path: str, *, resume: str | None = None) -> None:
    cfg = load_config(config_path)

    dist_env = setup_distributed("cuda")

    try:
        components = assemble_training_components(
            cfg,
            rank=dist_env.rank,
            world_size=dist_env.world_size,
            is_distributed=dist_env.is_distributed,
            is_main=dist_env.is_main,
        )

        model = components["model"].to(dist_env.device)
        trainer_cfg = components["trainer_cfg"]

        if trainer_cfg.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)

        if dist_env.is_distributed:
            model = DDP(
                model,
                device_ids=[dist_env.local_rank] if dist_env.device.type == "cuda" else None,
                output_device=dist_env.local_rank if dist_env.device.type == "cuda" else None,
            )

        optimizer = build_optimizer(model, components["optimizer_cfg"])
        scheduler = build_scheduler(optimizer, components["scheduler_cfg"])

        components["model"] = model
        components["optimizer"] = optimizer
        components["scheduler"] = scheduler

        trainer = Trainer.from_components(components, dist_env=dist_env)

        if resume is not None:
            trainer.load_checkpoint(resume)

        state = trainer.train()

        if dist_env.is_main:
            print("\nTraining finished.")
            print(f"step={state.step}")
            print(f"best_val_loss={state.best_val_loss}")
            print(f"last_train_loss={state.last_train_loss}")
            print(f"elapsed_seconds={state.elapsed_seconds}")

    finally:
        cleanup_distributed()

def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, required=True)
    parser.add_argument("--resume", default=None, help="path to checkpoint .pt file")
    args = parser.parse_args()
    main(config_path=args.config_path, resume=args.resume)


if __name__ == "__main__":
    cli()