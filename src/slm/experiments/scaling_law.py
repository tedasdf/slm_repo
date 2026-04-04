from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import wandb

from .base import BaseExperiment
from .callback import ExternalWandBCallback
from src.slm.training import RunConfig
from src.slm.training.builders import build_model, build_trainer


@dataclass
class ScalingLawExperimentConfig:
    compute_start: float
    compute_end: float
    compute_step: float

    ratio_start: float
    ratio_end: float
    ratio_step: float

    layers_list: tuple[int, ...]
    dims_list: tuple[int, ...]

    seed_values: tuple[int, ...] = (42,)
    train_flops_coeff: float = 6.0
    prefer_under_target: bool = True

    experiment_name: str = "scaling_law_fixed_compute"
    experiment_type: str = "scaling_law_fixed_compute"
    tags: list[str] = field(default_factory=list)


class ScalingLawExperiment(BaseExperiment):
    def __init__(self, base_cfg: RunConfig, experiment_cfg: ScalingLawExperimentConfig) -> None:
        super().__init__(base_cfg)

        self.experiment_cfg = experiment_cfg
        self._param_count_cache: dict[tuple[int, int, int, int], int] = {}

        self.compute_budgets = [
            round(experiment_cfg.compute_start + i * experiment_cfg.compute_step, 12)
            for i in range(
                int((experiment_cfg.compute_end - experiment_cfg.compute_start) / experiment_cfg.compute_step) + 1
            )
        ]
        self.data_param_ratios = [
            round(experiment_cfg.ratio_start + i * experiment_cfg.ratio_step, 12)
            for i in range(
                int((experiment_cfg.ratio_end - experiment_cfg.ratio_start) / experiment_cfg.ratio_step) + 1
            )
        ]

        self.legal_num_layers = list(experiment_cfg.layers_list)
        self.legal_model_dims = list(experiment_cfg.dims_list)

        self.candidate_models = self._build_candidate_models()

    def _infer_head_dim(self) -> int:
        attn = self.base_cfg.model.attention

        if getattr(attn, "head_dim", None) is not None:
            return int(attn.head_dim)

        num_heads = getattr(attn, "num_heads", None)
        model_dim = getattr(self.base_cfg.model, "model_dim", None)

        if num_heads is not None and model_dim is not None:
            num_heads = int(num_heads)
            model_dim = int(model_dim)
            if num_heads > 0 and model_dim % num_heads == 0:
                return model_dim // num_heads

        return 64

    def _build_candidate_models(self) -> list[dict[str, int]]:
        candidates: list[dict[str, int]] = []
        head_dim = self._infer_head_dim()

        for num_layers in self.legal_num_layers:
            for model_dim in self.legal_model_dims:
                if model_dim % head_dim != 0:
                    continue

                num_heads = model_dim // head_dim
                if num_heads <= 0:
                    continue

                candidates.append(
                    {
                        "num_layers": int(num_layers),
                        "model_dim": int(model_dim),
                        "num_heads": int(num_heads),
                        "head_dim": int(head_dim),
                    }
                )

        if not candidates:
            raise ValueError("No valid candidate models were generated.")

        return candidates

    def make_sweep_config(self) -> dict[str, Any]:
        return {
            "method": "grid",
            "metric": {
                "name": "eval/best_val_loss",
                "goal": "minimize",
            },
            "parameters": {
                "experiment_name": {
                    "values": [self.experiment_cfg.experiment_name]
                },
                "compute_budget": {
                    "values": self.compute_budgets
                },
                "data_param_ratio": {
                    "values": self.data_param_ratios
                },
                "seed": {
                    "values": list(self.experiment_cfg.seed_values)
                },
            },
        }

    def _resolve_targets_from_compute(
        self,
        *,
        compute_budget: float,
        data_param_ratio: float,
    ) -> dict[str, float]:
        if compute_budget <= 0:
            raise ValueError("compute_budget must be > 0")
        if data_param_ratio <= 0:
            raise ValueError("data_param_ratio must be > 0")

        k = float(self.experiment_cfg.train_flops_coeff)

        # C ~= k * N * D and D/N = r
        # => N = sqrt(C / (k*r)), D = r*N
        target_params = math.sqrt(compute_budget / (k * data_param_ratio))
        target_train_tokens = data_param_ratio * target_params

        return {
            "target_params": float(target_params),
            "target_train_tokens": float(target_train_tokens),
        }

    def _estimate_param_count(
        self,
        *,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        key = (int(num_layers), int(model_dim), int(num_heads), int(head_dim))
        cached = self._param_count_cache.get(key)
        if cached is not None:
            return cached

        cfg = deepcopy(self.base_cfg)
        cfg.model.num_layers = int(num_layers)
        cfg.model.model_dim = int(model_dim)
        cfg.model.attention.num_heads = int(num_heads)
        cfg.model.attention.head_dim = int(head_dim)

        model = build_model(cfg.model)
        n_params = sum(p.numel() for p in model.parameters())

        self._param_count_cache[key] = int(n_params)
        return int(n_params)

    def _choose_candidate_model(self, *, target_params: float) -> dict[str, Any]:
        scored_candidates: list[dict[str, Any]] = []

        for candidate in self.candidate_models:
            actual_params = self._estimate_param_count(
                num_layers=candidate["num_layers"],
                model_dim=candidate["model_dim"],
                num_heads=candidate["num_heads"],
                head_dim=candidate["head_dim"],
            )

            rel_error = abs(actual_params - target_params) / max(target_params, 1.0)

            scored_candidates.append(
                {
                    **candidate,
                    "actual_params": int(actual_params),
                    "param_rel_error": float(rel_error),
                }
            )

        if self.experiment_cfg.prefer_under_target:
            under = [c for c in scored_candidates if c["actual_params"] <= target_params]
            if under:
                scored_candidates = under

        best = min(scored_candidates, key=lambda c: c["param_rel_error"])
        return best

    def _tokens_per_step(self, cfg: RunConfig) -> int:
        trainer = cfg.trainer

        if hasattr(trainer, "effective_train_batch_tokens"):
            value = int(trainer.effective_train_batch_tokens)
            if value > 0:
                return value

        if hasattr(trainer, "micro_batch_tokens_per_rank"):
            micro = int(trainer.micro_batch_tokens_per_rank)
            grad_accum = int(getattr(trainer, "grad_accum_steps", 1))
            world_size = int(getattr(cfg.run, "world_size", 1)) if hasattr(cfg, "run") else 1
            value = micro * grad_accum * world_size
            if value > 0:
                return value

        batch_size = getattr(cfg.data, "batch_size", None)
        seq_len = getattr(cfg.trainer, "train_seq_len", None)
        if seq_len is None:
            seq_len = getattr(cfg.model, "max_seq_len", None)

        if batch_size is not None and seq_len is not None:
            value = int(batch_size) * int(seq_len)
            if value > 0:
                return value

        raise ValueError(
            "Could not infer tokens_per_step. "
            "Expose trainer.effective_train_batch_tokens or equivalent."
        )

    def _resolve_run_limits(
        self,
        *,
        cfg: RunConfig,
        compute_budget: float,
        actual_params: int,
    ) -> dict[str, int]:
        k = float(self.experiment_cfg.train_flops_coeff)

        actual_target_train_tokens = int(compute_budget / (k * max(actual_params, 1)))
        actual_target_train_tokens = max(actual_target_train_tokens, 1)

        tokens_per_step = self._tokens_per_step(cfg)
        max_steps = max(1, math.ceil(actual_target_train_tokens / tokens_per_step))

        return {
            "actual_target_train_tokens": int(actual_target_train_tokens),
            "tokens_per_step": int(tokens_per_step),
            "max_steps": int(max_steps),
        }

    def _build_run_tags(
        self,
        *,
        compute_budget: float,
        data_param_ratio: float,
        cfg: RunConfig,
    ) -> list[str]:
        base_tags = list(getattr(self.base_cfg.logging, "wandb_tags", []))
        experiment_tags = list(self.experiment_cfg.tags)

        resolved_tags = [
            f"budget:{compute_budget:.3e}",
            f"ratio:{data_param_ratio:.6g}",
            f"layers:{cfg.model.num_layers}",
            f"dim:{cfg.model.model_dim}",
            f"heads:{cfg.model.attention.num_heads}",
        ]

        return base_tags + experiment_tags + resolved_tags

    def apply_overrides(
        self,
        cfg: RunConfig,
        sweep_cfg: Any,
    ) -> tuple[RunConfig, dict[str, Any], dict[str, Any]]:
        cfg = deepcopy(cfg)

        experiment_name = str(
            getattr(sweep_cfg, "experiment_name", self.experiment_cfg.experiment_name)
        )

        compute_budget = float(sweep_cfg.compute_budget)
        data_param_ratio = float(sweep_cfg.data_param_ratio)

        theoretical = self._resolve_targets_from_compute(
            compute_budget=compute_budget,
            data_param_ratio=data_param_ratio,
        )

        chosen = self._choose_candidate_model(
            target_params=theoretical["target_params"]
        )

        cfg.model.num_layers = int(chosen["num_layers"])
        cfg.model.model_dim = int(chosen["model_dim"])
        cfg.model.attention.num_heads = int(chosen["num_heads"])
        cfg.model.attention.head_dim = int(chosen["head_dim"])

        if hasattr(sweep_cfg, "seed"):
            seed = int(sweep_cfg.seed)
            if hasattr(cfg.data, "seed"):
                cfg.data.seed = seed
            if hasattr(cfg.run, "seed"):
                cfg.run.seed = seed

        limits = self._resolve_run_limits(
            cfg=cfg,
            compute_budget=compute_budget,
            actual_params=int(chosen["actual_params"]),
        )

        cfg.trainer.max_steps = int(limits["max_steps"])

        if hasattr(cfg.trainer, "target_train_tokens"):
            cfg.trainer.target_train_tokens = int(limits["actual_target_train_tokens"])

        if hasattr(cfg.logging, "use_wandb"):
            cfg.logging.use_wandb = False

        run_tags = self._build_run_tags(
            compute_budget=compute_budget,
            data_param_ratio=data_param_ratio,
            cfg=cfg,
        )

        resolved = {
            "compute_budget": float(compute_budget),
            "data_param_ratio": float(data_param_ratio),
            "theoretical_target_params": float(theoretical["target_params"]),
            "theoretical_target_train_tokens": float(theoretical["target_train_tokens"]),
            "resolved_num_layers": int(chosen["num_layers"]),
            "resolved_model_dim": int(chosen["model_dim"]),
            "resolved_num_heads": int(chosen["num_heads"]),
            "resolved_head_dim": int(chosen["head_dim"]),
            "actual_params": int(chosen["actual_params"]),
            "param_rel_error": float(chosen["param_rel_error"]),
            "actual_target_train_tokens": int(limits["actual_target_train_tokens"]),
            "tokens_per_step": int(limits["tokens_per_step"]),
            "max_steps": int(limits["max_steps"]),
        }

        run_metadata = {
            "experiment_name": experiment_name,
            "experiment_type": self.experiment_cfg.experiment_type,
            "tags": run_tags,
        }

        return cfg, resolved, run_metadata

    def train_one_run(self) -> None:
        base_tags = list(getattr(self.base_cfg.logging, "wandb_tags", []))

        with wandb.init(
            project=self.base_cfg.logging.wandb_project,
            tags=base_tags + self.experiment_cfg.tags,
        ) as run:
            cfg, resolved, run_metadata = self.apply_overrides(self.base_cfg, run.config)

            try:
                run.tags = tuple(run_metadata["tags"])
            except Exception:
                pass

            run.config.update(
                {
                    "experiment": {
                        "name": run_metadata["experiment_name"],
                        "type": run_metadata["experiment_type"],
                        "tags": run_metadata["tags"],
                        "definition": asdict(self.experiment_cfg),
                    },
                    "model": asdict(cfg.model),
                    "optimizer": asdict(cfg.optimizer),
                    "scheduler": asdict(cfg.scheduler),
                    "trainer": asdict(cfg.trainer),
                    "data": asdict(cfg.data),
                    "logging": asdict(cfg.logging),
                    "resolved": resolved,
                },
                allow_val_change=True,
            )

            trainer = build_trainer(
                cfg,
                extra_callbacks=[ExternalWandBCallback(run=run)],
            )
            trainer.train()

    def sweep_fixed_compute(self, count: int | None = None) -> str:
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

    def run(self, count: int | None = None) -> str:
        return self.sweep_fixed_compute(count=count)

    def analyze_results(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "ScalingLawExperiment.analyze_results() not implemented yet."
        )