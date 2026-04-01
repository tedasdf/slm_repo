from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from slm.experiments.baseline import BaselineExperiment, BaselineExperimentConfig
from slm.experiments.runner import run_experiment
from slm.training.config import TrainerConfig
from slm.training.data import DataLoaderConfig
from slm.data.lm_loader import build_dataloaders as build_token_dataloaders


@dataclass
class BaselineRunConfig:
    experiment: BaselineExperimentConfig = field(default_factory=BaselineExperimentConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataLoaderConfig = field(default_factory=DataLoaderConfig)


def load_config(config_path: str | Path) -> BaselineRunConfig:
    schema = OmegaConf.structured(BaselineRunConfig)
    loaded_cfg = OmegaConf.load(str(config_path))
    merged = OmegaConf.merge(schema, loaded_cfg)

    missing = OmegaConf.missing_keys(merged)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")

    cfg = OmegaConf.to_object(merged)
    if not isinstance(cfg, BaselineRunConfig):
        raise TypeError(f"Expected BaselineRunConfig, got {type(cfg)}")
    return cfg


def build_dataloaders(cfg: BaselineRunConfig):
    return build_token_dataloaders(cfg.data)

def main(config_path: str) -> None:
    cfg = load_config(config_path)

    train_loader, val_loader = build_dataloaders(cfg)

    experiment = BaselineExperiment(
        cfg=cfg.experiment,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    result = run_experiment(
        experiment=experiment,
        trainer_cfg=cfg.trainer,
    )

    print("\nTraining finished.")
    print(f"Best val loss: {result.state.best_val_loss}")
    print(f"Elapsed seconds: {result.state.elapsed_seconds}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    main(args.config)