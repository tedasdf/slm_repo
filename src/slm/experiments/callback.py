from __future__ import annotations

from typing import Any, Optional

from .callbacks import Callback  # use the same base Callback you already use


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class ExternalWandBCallback(Callback):
    """
    Uses an already-open wandb run.
    Does NOT call wandb.init() or finish().
    """

    def __init__(self, run: Any) -> None:
        self._run = run

    def on_step_end(
        self,
        trainer: Any,
        step_outputs: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._run is None:
            return

        state = trainer.state
        extra = state.extra or {}

        payload: dict[str, Any] = {
            "trainer/step": state.step,
        }

        if state.last_train_loss is not None:
            payload["train/loss"] = float(state.last_train_loss)

        for key in [
            "optimizer/lr",
            "diagnostics/grad_norm",
            "timing/elapsed_since_start_sec",
            "diagnostics/has_nan_or_inf_loss",
        ]:
            value = extra.get(key)
            if _is_number(value):
                payload[key] = float(value)

        if step_outputs:
            for k, v in step_outputs.items():
                if _is_number(v):
                    payload[f"step/{k}"] = float(v)

        self._run.log(payload, step=state.step)

    def on_eval_end(
        self,
        trainer: Any,
        eval_outputs: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._run is None:
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
            payload["eval/best_val_loss"] = float(state.best_val_loss)

        self._run.log(payload, step=state.step)

    def on_run_end(self, trainer: Any) -> None:
        if self._run is None:
            return

        if trainer.state.best_val_loss is not None:
            self._run.summary["best_val_loss"] = float(trainer.state.best_val_loss)
        if trainer.state.elapsed_seconds is not None:
            self._run.summary["elapsed_seconds"] = float(trainer.state.elapsed_seconds)

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        if self._run is None:
            return

        self._run.summary["failed"] = True
        self._run.summary["error_type"] = type(exc).__name__
        self._run.summary["error_message"] = str(exc)