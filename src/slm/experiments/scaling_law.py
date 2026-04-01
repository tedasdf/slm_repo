from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import wandb

from slm.experiments.base import BaseExperiment
from slm.training.builders import build_trainer
from slm.training.run_config import RunConfig


class ScalingLawExperiment(BaseExperiment):
    def __init__(self, base_cfg: RunConfig) -> None:
        super().__init__(base_cfg)

    def make_sweep_config(self) -> dict[str, Any]:
        """
        Flat parameter names make wandb sweeps easier to work with.
        Adjust the values however you want.
        """
        return {
            "method": "grid",
            "metric": {
                "name": "eval/best_val_loss",
                "goal": "minimize",
            },
            "parameters": {
                "experiment_name": {
                    "values": ["scaling_law"]
                },
                "num_layers": {
                    "values": [6, 8, 10]
                },
                "model_dim": {
                    "values": [384, 512, 640]
                },
                "attention_type": {
                    "values": ["baseline"]
                },
                "compute_budget": {
                    "values": ["1.25e16", "2.50e16"]
                },
                "seed": {
                    "values": [42]
                },
            },
        }

    def apply_overrides(self, cfg: RunConfig, sweep_cfg: Any) -> RunConfig:
        """
        Takes the base run config and applies wandb sweep overrides.
        """
        cfg = deepcopy(cfg)

        # experiment metadata
        if hasattr(sweep_cfg, "experiment_name"):
            cfg.experiment.experiment_name = str(sweep_cfg.experiment_name)

        cfg.experiment.experiment_type = "scaling_law"

        # model overrides
        if hasattr(sweep_cfg, "num_layers"):
            cfg.model.num_layers = int(sweep_cfg.num_layers)

        if hasattr(sweep_cfg, "model_dim"):
            cfg.model.model_dim = int(sweep_cfg.model_dim)

        if hasattr(sweep_cfg, "attention_type"):
            cfg.model.attention.attention_type = str(sweep_cfg.attention_type)

        # optimizer overrides
        if hasattr(sweep_cfg, "lr"):
            cfg.optimizer.lr = float(sweep_cfg.lr)

        # trainer overrides
        if hasattr(sweep_cfg, "max_steps"):
            cfg.trainer.max_steps = int(sweep_cfg.max_steps)

        # data overrides
        if hasattr(sweep_cfg, "batch_size"):
            cfg.data.batch_size = int(sweep_cfg.batch_size)

        if hasattr(sweep_cfg, "seed"):
            seed = int(sweep_cfg.seed)
            cfg.data.seed = seed

        # logging
        # IMPORTANT: the experiment owns wandb.init() for sweeps,
        # so disable trainer-level wandb callback for sweep runs.
        cfg.logging.use_wandb = False

        # tags / metadata
        if hasattr(sweep_cfg, "compute_budget"):
            cfg.experiment.tags.append(f"budget:{sweep_cfg.compute_budget}")

        cfg.experiment.tags.append(f"layers:{cfg.model.num_layers}")
        cfg.experiment.tags.append(f"dim:{cfg.model.model_dim}")
        cfg.experiment.tags.append(f"attn:{cfg.model.attention.attention_type}")

        return cfg

    def train_one_run(self) -> None:
        with wandb.init(
            project=self.base_cfg.logging.wandb_project,
            tags=self.base_cfg.logging.wandb_tags + self.base_cfg.experiment.tags,
        ) as run:
            cfg = self.apply_overrides(self.base_cfg, run.config)

            # Save the fully resolved run config into wandb
            run.config.update(
                {
                    "experiment": asdict(cfg.experiment),
                    "model": asdict(cfg.model),
                    "optimizer": asdict(cfg.optimizer),
                    "scheduler": asdict(cfg.scheduler),
                    "trainer": asdict(cfg.trainer),
                    "data": asdict(cfg.data),
                    "logging": asdict(cfg.logging),
                },
                allow_val_change=True,
            )

            trainer = build_trainer(cfg)
            state = trainer.train()

            summary = {
                "experiment/name": cfg.experiment.experiment_name,
                "experiment/type": cfg.experiment.experiment_type,
                "model/num_layers": cfg.model.num_layers,
                "model/model_dim": cfg.model.model_dim,
                "model/attention_type": cfg.model.attention.attention_type,
                "optimizer/lr": cfg.optimizer.lr,
                "trainer/max_steps": cfg.trainer.max_steps,
                "data/batch_size": cfg.data.batch_size,
                "train/final_loss": state.last_train_loss,
                "eval/best_val_loss": state.best_val_loss,
                "runtime/elapsed_seconds": state.elapsed_seconds,
                "data/train_tokens_seen": state.train_tokens_seen,
                "data/train_samples_seen": state.train_samples_seen,
            }

            # optional diagnostics if trainer/callbacks filled them
            for key in [
                "diagnostics/grad_norm",
                "diagnostics/has_nan_or_inf_loss",
                "timing/elapsed_since_start_sec",
            ]:
                value = state.extra.get(key)
                if value is not None:
                    summary[key] = value

            # carry sweep-specific metadata through
            if hasattr(run.config, "compute_budget"):
                summary["budget/compute_target"] = run.config.compute_budget

            wandb.log(summary, step=state.step)

    def run(self, count: int | None = None) -> str:
        sweep_config = self.make_sweep_config()

        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=self.base_cfg.logging.wandb_project,
        )

        wandb.agent(
            sweep_id=sweep_id,
            function=self.train_one_run,
            count=count,
        )

        return sweep_id

    def analyze_results(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "ScalingLawExperiment.analyze_results() not implemented yet."
        )