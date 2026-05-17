from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from .experiments.scaling_law import (
    ScalingLawExperiment,
    ScalingLawExperimentConfig,
)
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


def load_experiment_config(
    experiment_path: str | Path,
) -> ScalingLawExperimentConfig:
    schema = OmegaConf.structured(ScalingLawExperimentConfig)
    loaded_cfg = OmegaConf.load(str(experiment_path))
    merged = OmegaConf.merge(schema, loaded_cfg)

    missing = OmegaConf.missing_keys(merged)
    if missing:
        raise ValueError(f"Missing experiment config fields: {sorted(missing)}")

    cfg = OmegaConf.to_object(merged)
    if not isinstance(cfg, ScalingLawExperimentConfig):
        raise TypeError(
            f"Expected ScalingLawExperimentConfig, got {type(cfg)}"
        )
    return cfg


def main(
    config_path: str,
    *,
    experiment: bool = False,
    experiment_path: str | None = None,
) -> None:
    cfg = load_config(config_path)

    if experiment:
        if experiment_path is None:
            raise ValueError(
                "--experiment_path is required when --experiment is set"
            )

        experiment_cfg = load_experiment_config(experiment_path)

        scaling_experiment = ScalingLawExperiment(
            base_cfg=cfg,
            experiment_cfg=experiment_cfg,
        )

        sweep_id = scaling_experiment.run()
        print(f"sweep_id={sweep_id}")
        return

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
        state = trainer.train()

        if dist_env.is_main:
            print("\nTraining finished.")
            print(f"step={state.step}")
            print(f"best_val_loss={state.best_val_loss}")
            print(f"last_train_loss={state.last_train_loss}")
            print(f"elapsed_seconds={state.elapsed_seconds}")

    finally:
        cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--experiment_path", type=str, default=None)
    parser.add_argument("--experiment", action="store_true")
    args = parser.parse_args()

    main(
        config_path=args.config_path,
        experiment=args.experiment,
        experiment_path=args.experiment_path,
    )