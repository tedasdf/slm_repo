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

from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class ScalingLawExperimentConfig:
    compute_start: float
    compute_num: int

    ratio_start: float
    ratio_end: float
    ratio_step: float

    layers_list: tuple[int, ...]
    dim_num_heads: dict[int, int]

    seed_values: tuple[int, ...] = (42,)
    train_flops_coeff: float = 6.0

    # Supported values: "exact", "kaplan", "hoffman"
    selection_param_mode: str = "kaplan"

    prefer_under_target: bool = True

    experiment_name: str = "scaling_law_fixed_compute"
    experiment_type: str = "scaling_law_fixed_compute"
    tags: list[str] = field(default_factory=list)

## computer budget ?  -> ratio decides tokens vs model size -> explicit depth --> find parameter N  



class ScalingLawExperiment(BaseExperiment):
    def __init__(self, base_cfg: RunConfig, experiment_cfg: ScalingLawExperimentConfig) -> None:
        super().__init__(base_cfg)

        self.experiment_cfg = experiment_cfg
        self._param_count_cache: dict[tuple[int, int, int, int], int] = {}

        self.compute_budgets = [
            round(experiment_cfg.compute_start * (2 ** i), 12)
            for i in range(
                experiment_cfg.compute_num 
            )
        ]
        self.data_param_ratios = [
            round(experiment_cfg.ratio_start + i * experiment_cfg.ratio_step, 12)
            for i in range(
                int((experiment_cfg.ratio_end - experiment_cfg.ratio_start) / experiment_cfg.ratio_step) + 1
            )
        ]

        self.legal_num_layers = list(experiment_cfg.layers_list)
        self.candidate_models = self._build_candidate_models()

    def _resolve_hidden_dim(self, *, model_dim: int) -> int:
        mlp_cfg = self.base_cfg.model.mlp

        hidden_dim = getattr(mlp_cfg, "hidden_dim", None)
        if hidden_dim is not None:
            return int(hidden_dim)

        mlp_mult = float(getattr(mlp_cfg, "mlp_mult", 4.0))
        return int(round(model_dim * mlp_mult))

    def _estimate_model_dim_from_target_params(
        self,
        *,
        target_params: float,
        target_depth: int,
        param_mode: str,
    ) -> float:
        """
        Estimate the ideal model_dim for a fixed depth before snapping
        to the nearest legal candidate width.
        """
        if param_mode not in {"exact", "kaplan", "hoffman"}:
            raise ValueError(
                f"Unsupported param_mode={param_mode!r}. "
                f"Use one of: 'exact', 'kaplan', 'hoffman'."
            )

        if target_depth <= 0:
            raise ValueError("target_depth must be > 0")
        if target_params <= 0:
            raise ValueError("target_params must be > 0")

        mlp_mult = float(getattr(self.base_cfg.model.mlp, "mlp_mult", 4.0))

        # N ~= 2 * L * (2 + mlp_mult) * d^2
        a = 2.0 * float(target_depth) * (2.0 + mlp_mult)

        if param_mode == "kaplan":
            return math.sqrt(target_params / max(a, 1e-12))

        # "exact" and "hoffman" both use total-parameter style estimate here
        vocab_size = int(self.base_cfg.model.vocab_size)
        tie_embeddings = bool(getattr(self.base_cfg.model, "tie_embeddings", True))
        b = float(vocab_size * (1 if tie_embeddings else 2))

        disc = b * b + 4.0 * a * target_params
        return max(1.0, (-b + math.sqrt(max(disc, 0.0))) / (2.0 * a))

    def _build_candidate_models(self) -> list[dict[str, int]]:
        candidates: list[dict[str, int]] = []

        for num_layers in self.legal_num_layers:
            for model_dim, num_heads in self.experiment_cfg.dim_num_heads.items():
                model_dim = int(model_dim)
                num_heads = int(num_heads)

                if num_heads <= 0:
                    continue

                if model_dim % num_heads != 0:
                    continue

                head_dim = model_dim // num_heads

                candidates.append(
                    {
                        "num_layers": int(num_layers),
                        "model_dim": model_dim,
                        "num_heads": num_heads,
                        "head_dim": head_dim,
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
                "num_layers": {
                    "values": list(self.experiment_cfg.layers_list)
                },
                "selection_param_mode": {
                    "values": [self.experiment_cfg.selection_param_mode]
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
        param_mode: str,
    ) -> dict[str, float]:
        if compute_budget <= 0:
            raise ValueError("compute_budget must be > 0")
        if data_param_ratio <= 0:
            raise ValueError("data_param_ratio must be > 0")

        if param_mode not in {"exact", "kaplan", "hoffman"}:
            raise ValueError(
                f"Unsupported param_mode={param_mode!r}. "
                f"Use one of: 'exact', 'kaplan', 'hoffman'."
            )

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
        mode: str,
    ) -> int:
        if mode not in {"exact", "kaplan", "hoffman"}:
            raise ValueError(
                f"Unsupported mode={mode!r}. "
                f"Use one of: 'exact', 'kaplan', 'hoffman'."
            )

        key = (
            mode,
            int(num_layers),
            int(model_dim),
            int(num_heads),
            int(head_dim),
        )

        cached = self._param_count_cache.get(key)
        if cached is not None:
            return cached

        if mode == "exact":
            cfg = deepcopy(self.base_cfg)
            cfg.model.num_layers = int(num_layers)
            cfg.model.model_dim = int(model_dim)
            cfg.model.attention.num_heads = int(num_heads)
            cfg.model.attention.head_dim = int(head_dim)

            if hasattr(cfg.model.attention, "num_kv_heads"):
                cfg.model.attention.num_kv_heads = int(num_heads)

            model = build_model(cfg.model)
            n_params = sum(p.numel() for p in model.parameters())

            self._param_count_cache[key] = int(n_params)
            return int(n_params)

        vocab_size = int(self.base_cfg.model.vocab_size)
        tie_embeddings = bool(getattr(self.base_cfg.model, "tie_embeddings", True))

        num_layers = int(num_layers)
        model_dim = int(model_dim)
        num_heads = int(num_heads)
        head_dim = int(head_dim)

        d_attn = num_heads * head_dim
        d_ff = self._resolve_hidden_dim(model_dim=model_dim)

        # Kaplan-style non-embedding proxy
        non_embedding = 2 * model_dim * num_layers * (2 * d_attn + d_ff)

        if mode == "kaplan":
            n_params = int(non_embedding)
        else:  # hoffman
            embed = vocab_size * model_dim
            deembed = 0 if tie_embeddings else vocab_size * model_dim
            n_params = int(non_embedding + embed + deembed)

        self._param_count_cache[key] = int(n_params)
        return int(n_params)

    def _choose_candidate_model(
        self,
        *,
        target_params: float,
        target_depth: int,
        param_mode: str,
    ) -> dict[str, Any]:
        if param_mode not in {"exact", "kaplan", "hoffman"}:
            raise ValueError(
                f"Unsupported param_mode={param_mode!r}. "
                f"Use one of: 'exact', 'kaplan', 'hoffman'."
            )

        depth = int(target_depth)

        candidates = [
            c for c in self.candidate_models
            if c["num_layers"] == depth
        ]
        if not candidates:
            raise ValueError(f"No candidate models found for target_depth={depth}")

        target_model_dim_estimate = self._estimate_model_dim_from_target_params(
            target_params=target_params,
            target_depth=depth,
            param_mode=param_mode,
        )

        scored_candidates: list[dict[str, Any]] = []

        for candidate in candidates:
            actual_params = self._estimate_param_count(
                num_layers=candidate["num_layers"],
                model_dim=candidate["model_dim"],
                num_heads=candidate["num_heads"],
                head_dim=candidate["head_dim"],
                mode=param_mode,
            )

            param_rel_error = abs(actual_params - target_params) / max(target_params, 1.0)
            model_dim_rel_error = (
                abs(candidate["model_dim"] - target_model_dim_estimate)
                / max(target_model_dim_estimate, 1.0)
            )

            scored_candidates.append(
                {
                    **candidate,
                    "actual_params": int(actual_params),
                    "param_rel_error": float(param_rel_error),
                    "target_model_dim_estimate": float(target_model_dim_estimate),
                    "model_dim_rel_error": float(model_dim_rel_error),
                }
            )

        if self.experiment_cfg.prefer_under_target:
            under = [c for c in scored_candidates if c["actual_params"] <= target_params]
            if under:
                scored_candidates = under

        best = min(
            scored_candidates,
            key=lambda c: (c["param_rel_error"], c["model_dim_rel_error"]),
        )
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
            world_size = int(getattr(cfg.trainer, "world_size", 1)) if hasattr(cfg, "run") else 1
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
        selection_param_mode: str,
    ) -> list[str]:
        base_tags = list(getattr(self.base_cfg.logging, "wandb_tags", []))
        experiment_tags = list(self.experiment_cfg.tags)

        resolved_tags = [
            f"budget:{compute_budget:.3e}",
            f"ratio:{data_param_ratio:.6g}",
            f"layers:{cfg.model.num_layers}",
            f"dim:{cfg.model.model_dim}",
            f"heads:{cfg.model.attention.num_heads}",
            f"param_mode:{selection_param_mode}",
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
        target_depth = int(sweep_cfg.num_layers)

        selection_param_mode = str(
            getattr(sweep_cfg, "selection_param_mode", self.experiment_cfg.selection_param_mode)
        )
        if selection_param_mode not in {"exact", "kaplan", "hoffman"}:
            raise ValueError(
                f"Unsupported selection_param_mode={selection_param_mode!r}. "
                f"Use one of: 'exact', 'kaplan', 'hoffman'."
            )

        theoretical = self._resolve_targets_from_compute(
            compute_budget=compute_budget,
            data_param_ratio=data_param_ratio,
            param_mode=selection_param_mode,
        )

        chosen = self._choose_candidate_model(
            target_params=theoretical["target_params"],
            target_depth=target_depth,
            param_mode=selection_param_mode,
        )

        cfg.model.num_layers = int(chosen["num_layers"])
        cfg.model.model_dim = int(chosen["model_dim"])
        cfg.model.attention.num_heads = int(chosen["num_heads"])
        cfg.model.attention.head_dim = int(chosen["head_dim"])

        if hasattr(cfg.model.attention, "num_kv_heads"):
            cfg.model.attention.num_kv_heads = int(chosen["num_heads"])

        if hasattr(sweep_cfg, "seed"):
            seed = int(sweep_cfg.seed)
            if hasattr(cfg.data, "seed"):
                cfg.data.seed = seed

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

        actual_params_exact = self._estimate_param_count(
            num_layers=chosen["num_layers"],
            model_dim=chosen["model_dim"],
            num_heads=chosen["num_heads"],
            head_dim=chosen["head_dim"],
            mode="exact",
        )
        actual_params_kaplan = self._estimate_param_count(
            num_layers=chosen["num_layers"],
            model_dim=chosen["model_dim"],
            num_heads=chosen["num_heads"],
            head_dim=chosen["head_dim"],
            mode="kaplan",
        )
        actual_params_hoffman = self._estimate_param_count(
            num_layers=chosen["num_layers"],
            model_dim=chosen["model_dim"],
            num_heads=chosen["num_heads"],
            head_dim=chosen["head_dim"],
            mode="hoffman",
        )

        run_tags = self._build_run_tags(
            compute_budget=compute_budget,
            data_param_ratio=data_param_ratio,
            cfg=cfg,
            selection_param_mode=selection_param_mode,
        )

        resolved = {
            "compute_budget": float(compute_budget),
            "data_param_ratio": float(data_param_ratio),
            "selection_param_mode": selection_param_mode,

            "theoretical_target_params": float(theoretical["target_params"]),
            "theoretical_target_train_tokens": float(theoretical["target_train_tokens"]),
            "target_depth": int(target_depth),
            "target_model_dim_estimate": float(chosen["target_model_dim_estimate"]),

            "resolved_num_layers": int(chosen["num_layers"]),
            "resolved_model_dim": int(chosen["model_dim"]),
            "resolved_num_heads": int(chosen["num_heads"]),
            "resolved_head_dim": int(chosen["head_dim"]),

            "actual_params_selected_mode": int(chosen["actual_params"]),
            "actual_params_exact": int(actual_params_exact),
            "actual_params_kaplan": int(actual_params_kaplan),
            "actual_params_hoffman": int(actual_params_hoffman),

            "param_rel_error": float(chosen["param_rel_error"]),
            "model_dim_rel_error": float(chosen["model_dim_rel_error"]),

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

    def _build_run_tags(
        self,
        *,
        compute_budget: float,
        data_param_ratio: float,
        cfg: RunConfig,
        selection_param_mode: str,
    ) -> list[str]:
        base_tags = list(getattr(self.base_cfg.logging, "wandb_tags", []))
        experiment_tags = list(self.experiment_cfg.tags)

        resolved_tags = [
            f"budget:{compute_budget:.3e}",
            f"ratio:{data_param_ratio:.6g}",
            f"layers:{cfg.model.num_layers}",
            f"dim:{cfg.model.model_dim}",
            f"heads:{cfg.model.attention.num_heads}",
            f"param_mode:{selection_param_mode}",
        ]

        return base_tags + experiment_tags + resolved_tags

if __name__ == "__main__":
    from omegaconf import OmegaConf

    experiment_path = r"configs/experiment/scaling_law.yaml"
    config_path = r"configs/train/baseline.yaml"

    schema = OmegaConf.structured(RunConfig)
    loaded_cfg = OmegaConf.load(str(config_path))
    merged = OmegaConf.merge(schema, loaded_cfg)
    cfg = OmegaConf.to_object(merged)

    schema = OmegaConf.structured(ScalingLawExperimentConfig)
    loaded_cfg = OmegaConf.load(str(experiment_path))
    merged = OmegaConf.merge(schema, loaded_cfg)
    experiment_cfg = OmegaConf.to_object(merged)

    print(experiment_cfg)

    scaling_experiment = ScalingLawExperiment(
        base_cfg=cfg,
        experiment_cfg=experiment_cfg,
    )

    seen_architectures: dict[tuple[int, int, int], list[tuple[float, float, str]]] = {}

    for param_mode in ["exact", "kaplan", "hoffman"]:
        print(f"\n========== PARAM MODE: {param_mode} ==========")

        for compute_budget in scaling_experiment.compute_budgets:
            for data_param_ratio in scaling_experiment.data_param_ratios:
                for num_layers in scaling_experiment.legal_num_layers:
                    theoretical = scaling_experiment._resolve_targets_from_compute(
                        compute_budget=compute_budget,
                        data_param_ratio=data_param_ratio,
                        param_mode=param_mode,
                    )

                    chosen = scaling_experiment._choose_candidate_model(
                        target_params=theoretical["target_params"],
                        target_depth=num_layers,
                        param_mode=param_mode,
                    )

                    actual_exact = scaling_experiment._estimate_param_count(
                        num_layers=chosen["num_layers"],
                        model_dim=chosen["model_dim"],
                        num_heads=chosen["num_heads"],
                        head_dim=chosen["head_dim"],
                        mode="exact",
                    )
                    actual_kaplan = scaling_experiment._estimate_param_count(
                        num_layers=chosen["num_layers"],
                        model_dim=chosen["model_dim"],
                        num_heads=chosen["num_heads"],
                        head_dim=chosen["head_dim"],
                        mode="kaplan",
                    )
                    actual_hoffman = scaling_experiment._estimate_param_count(
                        num_layers=chosen["num_layers"],
                        model_dim=chosen["model_dim"],
                        num_heads=chosen["num_heads"],
                        head_dim=chosen["head_dim"],
                        mode="hoffman",
                    )

                    assert chosen["num_layers"] == num_layers, (
                        f"Depth mismatch: requested {num_layers}, "
                        f"got {chosen['num_layers']}"
                    )

                    arch_key = (
                        chosen["num_layers"],
                        chosen["model_dim"],
                        chosen["num_heads"],
                    )
                    seen_architectures.setdefault(arch_key, []).append(
                        (compute_budget, data_param_ratio, param_mode)
                    )

                    print(
                        f"mode={param_mode:8s} "
                        f"compute={compute_budget:.3e} "
                        f"ratio={data_param_ratio:6.2f} "
                        f"depth={num_layers:2d} "
                        f"target_params={theoretical['target_params']:10.0f} "
                        f"chosen_dim={chosen['model_dim']:4d} "
                        f"heads={chosen['num_heads']:2d} "
                        f"head_dim={chosen['head_dim']:3d} "
                        f"selected_params={chosen['actual_params']:10d} "
                        f"exact={actual_exact:10d} "
                        f"kaplan={actual_kaplan:10d} "
                        f"hoffman={actual_hoffman:10d} "
                        f"rel_error={chosen['param_rel_error']:.4f}"
                    )

    from collections import defaultdict

    # arch_key -> list of runs
    # each run = (compute_budget, data_param_ratio, param_mode)
    duplicates_by_mode = defaultdict(list)
    unique_by_mode = defaultdict(list)

    for arch, runs in seen_architectures.items():
        for compute_budget, data_param_ratio, param_mode in runs:
            pass

        if len(runs) > 1:
            # store under each mode separately
            runs_by_mode = defaultdict(list)
            for compute_budget, data_param_ratio, param_mode in runs:
                runs_by_mode[param_mode].append((compute_budget, data_param_ratio))

            for param_mode, mode_runs in runs_by_mode.items():
                if len(mode_runs) > 1:
                    duplicates_by_mode[param_mode].append((arch, mode_runs))
                else:
                    unique_by_mode[param_mode].append((arch, mode_runs))
        else:
            compute_budget, data_param_ratio, param_mode = runs[0]
            unique_by_mode[param_mode].append((arch, [(compute_budget, data_param_ratio)]))


    with open("duplicate_architectures.txt", "w", encoding="utf-8") as f:
        for param_mode in ["exact", "kaplan", "hoffman"]:
            f.write(f"========== MODE: {param_mode} ==========\n")
            items = duplicates_by_mode.get(param_mode, [])
            if not items:
                f.write("No duplicate architectures.\n\n")
                continue

            for arch, mode_runs in items:
                f.write(f"arch={arch} used by {len(mode_runs)} runs\n")
                for compute_budget, data_param_ratio in sorted(mode_runs):
                    f.write(
                        f"  compute={compute_budget:.3e}, "
                        f"ratio={data_param_ratio:.2f}\n"
                    )
            f.write("\n")


    with open("unique_training_points.txt", "w", encoding="utf-8") as f:
        for param_mode in ["exact", "kaplan", "hoffman"]:
            f.write(f"========== MODE: {param_mode} ==========\n")
            items = unique_by_mode.get(param_mode, [])
            if not items:
                f.write("No unique points.\n\n")
                continue

            for arch, mode_runs in items:
                compute_budget, data_param_ratio = mode_runs[0]
                f.write(
                    f"arch={arch}, "
                    f"compute={compute_budget:.3e}, "
                    f"ratio={data_param_ratio:.2f}\n"
                )
            f.write("\n")