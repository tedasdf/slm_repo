from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import wandb


from src.slm.training import RunConfig
from src.slm.training.builders import build_model, build_trainer

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ScalingLawExperimentConfig:
    compute_list: tuple[float, ...]

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

    max_param_rel_error: float = 0.15
    min_steps: int = 1
    tags: list[str] = field(default_factory=list)


## Potential TODO add smoke test before 

class ScalingLawExperiment():
    def __init__(self, base_cfg: RunConfig, experiment_cfg):
        self.base_cfg = base_cfg

        self.experiment_cfg = experiment_cfg
        self.compute_budgets = sorted(list(experiment_cfg.compute_list))

        num_points = int((experiment_cfg.ratio_end - experiment_cfg.ratio_start) / experiment_cfg.ratio_step) + 1
        self.data_param_ratios = [
            round(experiment_cfg.ratio_start + i * experiment_cfg.ratio_step, 12)
            for i in range(num_points)
        ]

        self.legal_num_layers = list(experiment_cfg.layers_list)
        self.candidate_models = self._build_candidate_models()

    ### finding the best model globally
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

    def _choose_best_width_for_depth(
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

    def _choose_best_architectures(
        self,
        *,
        target_params: float,
        param_mode: str,
        return_per_depth: bool = True,
    ) -> dict[int, dict[str, Any]] | dict[str, Any]:
        if param_mode not in {"exact", "kaplan", "hoffman"}:
            raise ValueError(
                f"Unsupported param_mode={param_mode!r}. "
                f"Use one of: 'exact', 'kaplan', 'hoffman'."
            )

        best_by_depth: dict[int, dict[str, Any]] = {}

        for depth in self.legal_num_layers:
            best_by_depth[int(depth)] = self._choose_best_width_for_depth(
                target_params=target_params,
                target_depth=int(depth),
                param_mode=param_mode,
            )

        if return_per_depth:
            return best_by_depth

        best_overall = min(
            best_by_depth.values(),
            key=lambda c: (c["param_rel_error"], c["model_dim_rel_error"]),
        )
        return best_overall

    def _estimate_model_dim_from_target_params(
        self,
        *,
        target_params: float,
        target_depth: int,
        param_mode: str,
    ) -> float:
        if param_mode not in {"exact", "kaplan", "hoffman"}:
            raise ValueError(
                f"Unsupported param_mode={param_mode!r}. "
                f"Use one of: 'exact', 'kaplan', 'hoffman'."
            )

        if target_params <= 0:
            raise ValueError("target_params must be > 0")
        if target_depth <= 0:
            raise ValueError("target_depth must be > 0")

        model_cfg = self.base_cfg.model
        mlp_mult = float(getattr(model_cfg.mlp, "mlp_mult", 4.0))
        vocab_size = int(model_cfg.vocab_size)
        tie_embeddings = bool(getattr(model_cfg, "tie_embeddings", True))

        # Rough transformer body estimate:
        # N ≈ 2 * L * (2 + mlp_mult) * d^2
        a = 2.0 * float(target_depth) * (2.0 + mlp_mult)

        if param_mode == "kaplan":
            return math.sqrt(target_params / a)

        # For "exact" and "hoffman", include embedding params approximately:
        # N ≈ a*d^2 + b*d
        b = float(vocab_size * (1 if tie_embeddings else 2))

        disc = b * b + 4.0 * a * target_params
        return max(1.0, (-b + math.sqrt(disc)) / (2.0 * a))

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

        model_cfg = self.base_cfg.model
        vocab_size = int(model_cfg.vocab_size)
        tie_embeddings = bool(getattr(model_cfg, "tie_embeddings", True))
        mlp_mult = float(getattr(model_cfg.mlp, "mlp_mult", 4.0))

        d_model = int(model_dim)
        L = int(num_layers)
        d_attn = int(num_heads) * int(head_dim)
        d_ff = int(round(d_model * mlp_mult))

        # Transformer body only
        non_embedding = 2 * d_model * L * (2 * d_attn + d_ff)

        if mode == "kaplan":
            return int(non_embedding)

        # "hoffman" and simplified "exact"
        embed = vocab_size * d_model
        deembed = 0 if tie_embeddings else vocab_size * d_model

        return int(non_embedding + embed + deembed)
    
    def inspect_overlapping_architectures(
        self,
        *,
        param_mode: str = "kaplan",
    ) -> dict[str, Any]:
        if param_mode not in {"exact", "kaplan", "hoffman"}:
            raise ValueError(
                f"Unsupported param_mode={param_mode!r}. "
                f"Use one of: 'exact', 'kaplan', 'hoffman'."
            )

        overlaps_by_budget: dict[
            float,
            dict[tuple[int, int, int], list[dict[str, float | int]]]
        ] = {}

        max_theoretical: dict[str, Any] | None = None
        max_actual: dict[str, Any] | None = None

        k = float(self.experiment_cfg.train_flops_coeff)

        for compute_budget in self.compute_budgets:
            chosen_by_arch: dict[
                tuple[int, int, int],
                list[dict[str, float | int]]
            ] = {}

            for data_param_ratio in self.data_param_ratios:
                theoretical = self._resolve_targets_from_compute(
                    compute_budget=compute_budget,
                    data_param_ratio=data_param_ratio,
                    param_mode=param_mode,
                )

                best_overall = self._choose_best_architectures(
                    target_params=theoretical["target_params"],
                    param_mode=param_mode,
                    return_per_depth=False,
                )

                actual_target_train_tokens = compute_budget / (
                    k * max(best_overall["actual_params"], 1)
                )

                if (
                    max_theoretical is None
                    or theoretical["target_train_tokens"] > max_theoretical["tokens"]
                ):
                    max_theoretical = {
                        "tokens": float(theoretical["target_train_tokens"]),
                        "compute_budget": float(compute_budget),
                        "data_param_ratio": float(data_param_ratio),
                        "architecture": (
                            int(best_overall["num_layers"]),
                            int(best_overall["model_dim"]),
                            int(best_overall["num_heads"]),
                        ),
                        "actual_params": int(best_overall["actual_params"]),
                    }

                if (
                    max_actual is None
                    or actual_target_train_tokens > max_actual["tokens"]
                ):
                    max_actual = {
                        "tokens": float(actual_target_train_tokens),
                        "compute_budget": float(compute_budget),
                        "data_param_ratio": float(data_param_ratio),
                        "architecture": (
                            int(best_overall["num_layers"]),
                            int(best_overall["model_dim"]),
                            int(best_overall["num_heads"]),
                        ),
                        "actual_params": int(best_overall["actual_params"]),
                    }

                arch_key = (
                    int(best_overall["num_layers"]),
                    int(best_overall["model_dim"]),
                    int(best_overall["num_heads"]),
                )

                chosen_by_arch.setdefault(arch_key, []).append(
                    {
                        "data_param_ratio": float(data_param_ratio),
                        "target_params": float(theoretical["target_params"]),
                        "target_train_tokens": float(theoretical["target_train_tokens"]),
                        "actual_params": int(best_overall["actual_params"]),
                        "actual_target_train_tokens": float(actual_target_train_tokens),
                        "param_rel_error": float(best_overall["param_rel_error"]),
                    }
                )

            overlaps_only = {
                arch: runs
                for arch, runs in chosen_by_arch.items()
                if len(runs) > 1
            }

            overlaps_by_budget[float(compute_budget)] = overlaps_only

        return {
            "overlaps_by_budget": overlaps_by_budget,
            "max_theoretical": max_theoretical,
            "max_actual": max_actual,
        }

    def debug_print_resolved_grid(self, param_mode: str = "kaplan") -> None:
        for compute_budget in self.compute_budgets:
            print(f"\n========== compute_budget={compute_budget:.3e} ==========")

            for data_param_ratio in self.data_param_ratios:
                theoretical = self._resolve_targets_from_compute(
                    compute_budget=compute_budget,
                    data_param_ratio=data_param_ratio,
                    param_mode=param_mode,
                )

                chosen = self._choose_best_architectures(
                    target_params=theoretical["target_params"],
                    param_mode=param_mode,
                    return_per_depth=False,
                )

                limits = self._resolve_run_limits(
                    cfg=self.base_cfg,
                    compute_budget=compute_budget,
                    actual_params=int(chosen["actual_params"]),
                )

                print(
                    f"ratio={data_param_ratio:.2f} | "
                    f"layers={chosen['num_layers']} | "
                    f"d_model={chosen['model_dim']} | "
                    f"heads={chosen['num_heads']} | "
                    f"head_dim={chosen['head_dim']} | "
                    f"target_params={theoretical['target_params']:.0f} | "
                    f"actual_params={chosen['actual_params']} | "
                    f"param_rel_error={chosen['param_rel_error']:.4f} | "
                    f"target_tokens={theoretical['target_train_tokens']:.0f} | "
                    f"actual_target_tokens={limits['actual_target_train_tokens']} | "
                    f"tokens_per_step={limits['tokens_per_step']} | "
                    f"max_steps={limits['max_steps']}"
                )

    ### finding the best model globally

    ## config training into ###
    def _tokens_per_step(self, cfg: RunConfig) -> int:
        batch_size = int(cfg.data.batch_size)
        seq_len = int(cfg.data.seq_len)
        grad_accum_steps = int(cfg.trainer.grad_accum_steps)
        return batch_size * seq_len * grad_accum_steps

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
            "actual_target_train_tokens": actual_target_train_tokens,
            "tokens_per_step": tokens_per_step,
            "max_steps": max_steps,
        }

    def make_sweep_config(self) -> dict[str, Any]:
        return {
            "method": "grid",
            "metric": {
                "name": "eval/best_val_loss",
                "goal": "minimize",
            },
            "parameters": {
                "compute_budget": {
                    "values": self.compute_budgets,
                },
                "data_param_ratio": {
                    "values": self.data_param_ratios,
                },
                "selection_param_mode": {
                    "values": [self.experiment_cfg.selection_param_mode],
                },
                "seed": {
                    "values": list(self.experiment_cfg.seed_values),
                },
            },
        }
    
    def apply_overrides(self, cfg: RunConfig, sweep_cfg: Any):
        cfg = deepcopy(cfg)

        compute_budget = float(sweep_cfg.compute_budget)
        data_param_ratio = float(sweep_cfg.data_param_ratio)
        param_mode = str(getattr(sweep_cfg, "selection_param_mode", "kaplan"))

        theoretical = self._resolve_targets_from_compute(
            compute_budget=compute_budget,
            data_param_ratio=data_param_ratio,
            param_mode=param_mode,
        )

        chosen = self._choose_best_architectures(
            target_params=theoretical["target_params"],
            param_mode=param_mode,
            return_per_depth=False,
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
        cfg.trainer.target_train_tokens = int(limits["actual_target_train_tokens"])

        if hasattr(cfg.data, "train_token_budget"):
            cfg.data.train_token_budget = int(limits["actual_target_train_tokens"])

        resolved = {
            "compute_budget": compute_budget,
            "data_param_ratio": data_param_ratio,
            "selection_param_mode": param_mode,
            "theoretical_target_params": theoretical["target_params"],
            "theoretical_target_train_tokens": theoretical["target_train_tokens"],
            "resolved_num_layers": chosen["num_layers"],
            "resolved_model_dim": chosen["model_dim"],
            "resolved_num_heads": chosen["num_heads"],
            "resolved_head_dim": chosen["head_dim"],
            "actual_params": chosen["actual_params"],
            "param_rel_error": chosen["param_rel_error"],
            "model_dim_rel_error": chosen["model_dim_rel_error"],
            "actual_target_train_tokens": limits["actual_target_train_tokens"],
            "tokens_per_step": limits["tokens_per_step"],
            "max_steps": limits["max_steps"],
        }

        return cfg, resolved
    ## config training into ###

    def train_one_run(self) -> None:
        max_param_rel_error = float(
            getattr(self.experiment_cfg, "max_param_rel_error", 0.15)
        )
        min_steps = int(
            getattr(self.experiment_cfg, "min_steps", 1)
        )

        with wandb.init(
            project=self.base_cfg.logging.wandb_project,
            tags=list(self.experiment_cfg.tags),
        ) as run:
            cfg, resolved = self.apply_overrides(self.base_cfg, run.config)

            param_rel_error = float(resolved["param_rel_error"])
            max_steps = int(resolved["max_steps"])

            skip_reasons: list[str] = []

            if param_rel_error > max_param_rel_error:
                skip_reasons.append(
                    f"param_rel_error={param_rel_error:.4f} > {max_param_rel_error:.4f}"
                )

            if max_steps < min_steps:
                skip_reasons.append(
                    f"max_steps={max_steps} < min_steps={min_steps}"
                )

            should_skip = len(skip_reasons) > 0

            run.config.update(
                {
                    "experiment": {
                        "name": self.experiment_cfg.experiment_name,
                        "type": self.experiment_cfg.experiment_type,
                        "definition": asdict(self.experiment_cfg),
                    },
                    "resolved": resolved,
                    "selection": {
                        "should_skip": should_skip,
                        "skip_reasons": skip_reasons,
                        "max_param_rel_error": max_param_rel_error,
                        "min_steps": min_steps,
                    },
                },
                allow_val_change=True,
            )

            run.summary["should_skip"] = should_skip
            run.summary["param_rel_error"] = param_rel_error
            run.summary["max_steps"] = max_steps
            run.summary["actual_params"] = int(resolved["actual_params"])
            run.summary["actual_target_train_tokens"] = int(
                resolved["actual_target_train_tokens"]
            )

            if should_skip:
                skip_reason_text = " | ".join(skip_reasons)
                run.summary["skip_reason"] = skip_reason_text
                print(
                    "[SKIP] "
                    f"compute={resolved['compute_budget']:.3e}, "
                    f"ratio={resolved['data_param_ratio']:.2f}, "
                    f"arch=({resolved['resolved_num_layers']}, "
                    f"{resolved['resolved_model_dim']}, "
                    f"{resolved['resolved_num_heads']}), "
                    f"reason={skip_reason_text}"
                )
                return

            trainer = build_trainer(cfg)
            trainer.train()

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
    scaling_experiment.debug_print_resolved_grid(param_mode="kaplan")

    # seen_architectures: dict[tuple[int, int, int], list[tuple[float, float, str]]] = {}
    
    # result = scaling_experiment.inspect_overlapping_architectures(param_mode="kaplan")

    # overlaps = result["overlaps_by_budget"]
    # max_theoretical = result["max_theoretical"]
    # max_actual = result["max_actual"]

    # print("\n===== HIGHEST THEORETICAL TOKENS =====")
    # print(max_theoretical)

    # print("\n===== HIGHEST ACTUAL TOKENS =====")
    # print(max_actual)

    # for compute_budget, arch_dict in overlaps.items():
    #     print(f"\n========== compute_budget={compute_budget:.3e} ==========")

    #     if not arch_dict:
    #         print("No overlapping architectures within this compute budget.")
    #         continue

    #     for arch, runs in arch_dict.items():
    #         print(f"arch={arch} used by {len(runs)} ratios")
    #         for run in runs:
    #             print(
    #                 f"  ratio={run['data_param_ratio']:.2f} | "
    #                 f"target_params={run['target_params']:.0f} | "
    #                 f"target_tokens={run['target_train_tokens']:.0f} | "
    #                 f"actual_target_tokens={run['actual_target_train_tokens']:.0f} | "
    #                 f"actual_params={run['actual_params']} | "
    #                 f"param_rel_error={run['param_rel_error']:.4f}"
    #             )
                
    # max_theoretical = None
    # max_actual = None
    # for compute_budget in scaling_experiment.compute_budgets:
    #     for data_param_ratio in scaling_experiment.data_param_ratios:
    #         theoretical = scaling_experiment._resolve_targets_from_compute(
    #             compute_budget=compute_budget,
    #             data_param_ratio=data_param_ratio,
    #             param_mode="kaplan",
    #         )

    #         best_overall = scaling_experiment._choose_best_architectures(
    #             target_params=theoretical["target_params"],
    #             param_mode="kaplan",
    #             return_per_depth=False,
    #         )

    #         k = scaling_experiment.experiment_cfg.train_flops_coeff
    #     actual_target_train_tokens = compute_budget / (k * best_overall["actual_params"])

    #     if max_theoretical is None or theoretical["target_train_tokens"] > max_theoretical["tokens"]:
    #         max_theoretical = {
    #             "tokens": theoretical["target_train_tokens"],
    #             "compute_budget": compute_budget,
    #             "data_param_ratio": data_param_ratio,
    #             "best_overall": best_overall,
    #         }

    #     if max_actual is None or actual_target_train_tokens > max_actual["tokens"]:
    #         max_actual = {
    #             "tokens": actual_target_train_tokens,
    #             "compute_budget": compute_budget,
    #             "data_param_ratio": data_param_ratio,
    #             "best_overall": best_overall,
    #         }

    # print("\n===== HIGHEST THEORETICAL TOKENS =====")
    # print(max_theoretical)

    # print("\n===== HIGHEST ACTUAL TOKENS =====")
    # print(max_actual)