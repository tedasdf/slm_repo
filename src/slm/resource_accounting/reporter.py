from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .estimator import ResourceEstimate
    from .budget import BudgetReport


class Reporter:
    """Handles all resource accounting output: W&B and local JSON.

    Designed to be used by ResourceAccountingCallback but can also be called
    standalone (e.g. from the preprocessing hook).

    W&B logging:
        Checks for an active wandb.run before logging — safe to call even if
        WandBCallback hasn't been added or wandb is not installed. All metrics
        are logged under the "resource/" namespace to stay separate from the
        existing training metrics (loss, lr, grad_norm, tok/s, etc.).

    JSON summary:
        Written to <checkpoint_dir>/resource_summary.json at run end.
        Contains preflight estimates, budget report, and a list of actual
        sampled metrics collected during training.
    """

    def __init__(
        self,
        *,
        log_to_wandb: bool = True,
        write_json_summary: bool = True,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
    ) -> None:
        self.log_to_wandb = log_to_wandb
        self.write_json_summary = write_json_summary
        self.checkpoint_dir = Path(checkpoint_dir)

        # Accumulates actual sampled metrics across training for the JSON summary.
        self._actual_samples: list[dict[str, Any]] = []

    # ── Preflight ─────────────────────────────────────────────────────────────

    def log_preflight(
        self,
        estimate: "ResourceEstimate",
        budget_report: "BudgetReport",
    ) -> None:
        """Called once before training starts.

        Prints the budget summary to stdout (always).
        Logs estimate fields to W&B as resource/est_* scalars (if enabled).
        """
        print(f"\n[resource_accounting] preflight — {budget_report.summary}")

        if estimate.num_params is not None:
            print(f"  params:          {estimate.num_params:,}")
        if estimate.param_mem_gb is not None:
            print(f"  param memory:    {estimate.param_mem_gb:.2f} GB")
        if estimate.activation_mem_gb is not None:
            print(f"  activation mem:  {estimate.activation_mem_gb:.2f} GB")
        if estimate.optimizer_mem_gb is not None:
            print(f"  optimizer mem:   {estimate.optimizer_mem_gb:.2f} GB")
        if estimate.total_mem_gb is not None:
            print(f"  total est. mem:  {estimate.total_mem_gb:.2f} GB")
        if estimate.flops_per_step is not None:
            print(f"  FLOPs/step:      {estimate.flops_per_step:.3e}")
        if estimate.est_total_hours is not None:
            print(f"  est. total time: {estimate.est_total_hours:.2f} h")
        print()

        if self.log_to_wandb:
            self._wandb_log_preflight(estimate, budget_report)

    def _wandb_log_preflight(
        self,
        estimate: "ResourceEstimate",
        budget_report: "BudgetReport",
    ) -> None:
        """Log preflight as W&B summary fields (not per-step metrics)."""
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        fields: dict[str, Any] = {
            "resource/fit_status": budget_report.status.value,
            "resource/budget_gb": budget_report.budget_gb,
        }

        if estimate.num_params is not None:
            fields["resource/est_num_params"] = estimate.num_params
        if estimate.param_mem_gb is not None:
            fields["resource/est_param_mem_gb"] = estimate.param_mem_gb
        if estimate.activation_mem_gb is not None:
            fields["resource/est_activation_mem_gb"] = estimate.activation_mem_gb
        if estimate.optimizer_mem_gb is not None:
            fields["resource/est_optimizer_mem_gb"] = estimate.optimizer_mem_gb
        if estimate.total_mem_gb is not None:
            fields["resource/est_total_mem_gb"] = estimate.total_mem_gb
        if estimate.flops_per_step is not None:
            fields["resource/est_flops_per_step"] = estimate.flops_per_step
        if estimate.flops_per_token is not None:
            fields["resource/est_flops_per_token"] = estimate.flops_per_token
        if estimate.est_step_time_sec is not None:
            fields["resource/est_step_time_sec"] = estimate.est_step_time_sec
        if estimate.est_total_hours is not None:
            fields["resource/est_total_hours"] = estimate.est_total_hours
        if budget_report.required_gb is not None:
            fields["resource/est_required_gb"] = budget_report.required_gb
        if budget_report.headroom_fraction is not None:
            fields["resource/est_memory_headroom_pct"] = budget_report.headroom_fraction * 100

        # Log as namespaced W&B metrics, not run.summary fields.
        wandb.log(fields, step=0)

    # ── Per-step actuals ──────────────────────────────────────────────────────

    def log_actuals(self, step: int, state_extra: dict[str, Any]) -> None:
        """Sample actual resource metrics from trainer.state.extra.

        Pulls the fields that trainer.py already populates:
          - gpu/memory_reserved_gb
          - throughput/tokens_per_sec
          - timing/elapsed_since_start_sec

        Called by ResourceAccountingCallback.on_step_end every
        sample_every_n_steps steps.
        """
        actuals: dict[str, Any] = {"step": step}

        mem = state_extra.get("gpu/memory_reserved_gb")
        tps = state_extra.get("throughput/tokens_per_sec")
        elapsed = state_extra.get("timing/elapsed_since_start_sec")

        if mem is not None:
            actuals["gpu/memory_reserved_gb"] = mem
        if tps is not None:
            actuals["throughput/tokens_per_sec"] = tps
        if elapsed is not None:
            actuals["timing/elapsed_since_start_sec"] = elapsed

        self._actual_samples.append(actuals)

        if self.log_to_wandb:
            self._wandb_log_actuals(step, actuals)

    def _wandb_log_actuals(self, step: int, actuals: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        payload: dict[str, Any] = {}
        if "gpu/memory_reserved_gb" in actuals:
            payload["resource/actual_gpu_mem_gb"] = actuals["gpu/memory_reserved_gb"]
        if "throughput/tokens_per_sec" in actuals:
            payload["resource/actual_tokens_per_sec"] = actuals["throughput/tokens_per_sec"]

        if payload:
            wandb.log(payload, step=step)

    # ── Run-end JSON summary ──────────────────────────────────────────────────

    def write_summary(
        self,
        estimate: "ResourceEstimate",
        budget_report: "BudgetReport",
    ) -> Path | None:
        """Write resource_summary.json to the checkpoint directory.

        Contains:
          - preflight estimates (all ResourceEstimate fields)
          - budget report (status, budget_gb, required_gb, headroom, summary)
          - list of actual sampled metrics collected during training
        """
        if not self.write_json_summary:
            return None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.checkpoint_dir / "resource_summary.json"

        summary = {
            "preflight": {
                "num_params": estimate.num_params,
                "param_mem_gb": estimate.param_mem_gb,
                "activation_mem_gb": estimate.activation_mem_gb,
                "optimizer_mem_gb": estimate.optimizer_mem_gb,
                "total_mem_gb": estimate.total_mem_gb,
                "flops_per_step": estimate.flops_per_step,
                "flops_per_token": estimate.flops_per_token,
                "est_step_time_sec": estimate.est_step_time_sec,
                "est_total_hours": estimate.est_total_hours,
            },
            "budget": {
                "status": budget_report.status.value,
                "budget_gb": budget_report.budget_gb,
                "required_gb": budget_report.required_gb,
                "headroom_fraction": budget_report.headroom_fraction,
                "summary": budget_report.summary,
            },
            "actuals": self._actual_samples,
        }

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[resource_accounting] summary written → {out_path}")
        return out_path