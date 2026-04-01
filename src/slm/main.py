from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

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


def main(config_path: str) -> None:
    cfg = load_config(config_path)

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
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)