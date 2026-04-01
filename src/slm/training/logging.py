from __future__ import annotations

import math
from typing import Any, Optional

from .callbacks import Callback


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if _is_number(x):
        x = float(x)
        if math.isfinite(x):
            return x
    return None


class PrintMetricsCallback(Callback):
    def __init__(self, prefix: str = "[train]") -> None:
        self.prefix = prefix

    def on_step_end(self, trainer: Any, step_outputs: Optional[dict[str, Any]] = None) -> None:
        state = trainer.state
        extra = state.extra

        loss = _to_float(state.last_train_loss)
        lr = _to_float(extra.get("optimizer/lr"))
        grad_norm = _to_float(extra.get("diagnostics/grad_norm"))
        elapsed = _to_float(extra.get("timing/elapsed_since_start_sec"))

        parts = [f"{self.prefix} step={state.step}"]

        if loss is not None:
            parts.append(f"loss={loss:.6f}")
        if lr is not None:
            parts.append(f"lr={lr:.6e}")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.4f}")
        if elapsed is not None:
            parts.append(f"elapsed={elapsed:.1f}s")

        print(" | ".join(parts))

    def on_eval_end(self, trainer: Any, eval_outputs: Optional[dict[str, Any]] = None) -> None:
        eval_outputs = eval_outputs or {}
        val_loss = _to_float(eval_outputs.get("val_loss"))
        is_best = bool(eval_outputs.get("is_best", False))

        parts = [f"[eval] step={trainer.state.step}"]
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.6f}")
        parts.append(f"is_best={is_best}")

        print(" | ".join(parts))

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        print(f"[error] step={trainer.state.step} | {type(exc).__name__}: {exc}")


class WandBCallback(Callback):
    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.enabled = enabled

        self._wandb = None
        self._run = None

    def on_run_start(self, trainer: Any) -> None:
        if not self.enabled:
            return

        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            tags=self.tags,
        )

    def on_step_end(self, trainer: Any, step_outputs: Optional[dict[str, Any]] = None) -> None:
        if not self.enabled or self._wandb is None:
            return

        state = trainer.state
        extra = state.extra

        payload: dict[str, Any] = {
            "trainer/step": state.step,
        }

        if state.last_train_loss is not None:
            payload["train/loss"] = state.last_train_loss

        for key in [
            "optimizer/lr",
            "diagnostics/grad_norm",
            "timing/elapsed_since_start_sec",
            "diagnostics/has_nan_or_inf_loss",
        ]:
            value = extra.get(key)
            if value is not None:
                payload[key] = value

        if step_outputs:
            for k, v in step_outputs.items():
                if _is_number(v):
                    payload[f"step/{k}"] = float(v)

        self._wandb.log(payload, step=state.step)

    def on_eval_end(self, trainer: Any, eval_outputs: Optional[dict[str, Any]] = None) -> None:
        if not self.enabled or self._wandb is None:
            return

        state = trainer.state
        eval_outputs = eval_outputs or {}

        payload: dict[str, Any] = {
            "trainer/step": state.step,
        }

        for k, v in eval_outputs.items():
            if _is_number(v):
                payload[f"eval/{k}"] = float(v)

        if state.best_val_loss is not None:
            payload["eval/best_val_loss"] = state.best_val_loss

        self._wandb.log(payload, step=state.step)

    def on_run_end(self, trainer: Any) -> None:
        if not self.enabled or self._run is None:
            return

        summary = {}
        if trainer.state.best_val_loss is not None:
            summary["best_val_loss"] = trainer.state.best_val_loss
        if trainer.state.elapsed_seconds is not None:
            summary["elapsed_seconds"] = trainer.state.elapsed_seconds

        for k, v in summary.items():
            self._run.summary[k] = v

        self._run.finish()
        self._run = None
        self._wandb = None

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        if not self.enabled or self._run is None:
            return
        self._run.summary["failed"] = True
        self._run.summary["error_type"] = type(exc).__name__
        self._run.summary["error_message"] = str(exc)