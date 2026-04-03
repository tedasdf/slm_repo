from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from .experiments.scaling_law import (
    ScalingLawExperiment,
    ScalingLawExperimentConfig,
)
from .training.builders import assemble_training_components
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

    components = assemble_training_components(cfg)
    trainer = Trainer.from_components(components)

    state = trainer.train()

    print("\nTraining finished.")
    print(f"step={state.step}")
    print(f"best_val_loss={state.best_val_loss}")
    print(f"last_train_loss={state.last_train_loss}")
    print(f"elapsed_seconds={state.elapsed_seconds}")


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