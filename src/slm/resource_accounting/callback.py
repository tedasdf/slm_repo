from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from ..training.callbacks import Callback
from .config import ResourceConfig
from .estimator import estimate_resources
from .budget import check_budget, FitStatus
from .reporter import Reporter

if TYPE_CHECKING:
    from ..model.config import ModelConfig
    from ..training.run_config import TrainerConfig


class ResourceAccountingCallback(Callback):
    """Preflight estimation + per-step resource sampling for the Trainer.

    Lifecycle:
      on_train_start  → run preflight (estimate + budget check + log)
                        optionally abort if OOM predicted
      on_step_end     → every sample_every_n_steps, read state.extra and
                        forward actual metrics to reporter
      on_run_end      → write resource_summary.json

    Usage — add to your callbacks list in builders.py or assemble_training_components:

        from slm.resource_accounting import ResourceAccountingCallback, ResourceConfig

        resource_cfg = ResourceConfig(gpu_tflops=312.0, gpu_mem_gb=80.0)
        callbacks = [
            ...,
            ResourceAccountingCallback(
                model_cfg=run_cfg.model,
                trainer_cfg=run_cfg.trainer,
                resource_cfg=resource_cfg,
            ),
        ]

    Or add ResourceConfig to RunConfig and wire it up in builders.py alongside
    the other callbacks (LoggingConfig → WandBCallback pattern).
    """

    def __init__(
        self,
        model_cfg: "ModelConfig",
        trainer_cfg: "TrainerConfig",
        resource_cfg: ResourceConfig | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg
        self.resource_cfg = resource_cfg or ResourceConfig()

        self.reporter = Reporter(
            log_to_wandb=self.resource_cfg.log_to_wandb,
            write_json_summary=self.resource_cfg.write_json_summary,
            checkpoint_dir=self.trainer_cfg.checkpoint_dir,
        )

        # Populated during on_train_start, used in on_run_end.
        self._estimate = None
        self._budget_report = None

    # ── Preflight ─────────────────────────────────────────────────────────────

    def on_train_start(self, trainer: Any) -> None:
        if not self.resource_cfg.run_preflight:
            return

        self._estimate = estimate_resources(
            model_cfg=self.model_cfg,
            trainer_cfg=self.trainer_cfg,
            resource_cfg=self.resource_cfg,
            world_size=trainer.world_size,
        )

        self._budget_report = check_budget(
            self._estimate,
            self.resource_cfg,
            world_size=trainer.world_size,
        )

        self.reporter.log_preflight(self._estimate, self._budget_report)

        if (
            self.resource_cfg.abort_on_oom
            and self._budget_report.status == FitStatus.OOM
        ):
            raise RuntimeError(
                f"[resource_accounting] Aborting: {self._budget_report.summary}"
            )

    # ── Per-step actuals ──────────────────────────────────────────────────────

    def on_step_end(
        self, trainer: Any, step_outputs: Optional[dict[str, Any]] = None
    ) -> None:
        step = trainer.state.step

        if step % self.resource_cfg.sample_every_n_steps != 0:
            return

        # trainer.state.extra already contains gpu/memory_reserved_gb and
        # throughput/tokens_per_sec — populated by trainer.py's main loop.
        self.reporter.log_actuals(step, trainer.state.extra)

    # ── Run-end summary ───────────────────────────────────────────────────────

    def on_run_end(self, trainer: Any) -> None:
        if self._estimate is None or self._budget_report is None:
            return

        self.reporter.write_summary(self._estimate, self._budget_report)