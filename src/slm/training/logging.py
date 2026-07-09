from __future__ import annotations

import math
import os
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
        if not getattr(trainer, "is_main", True):
            return
        state = trainer.state
        extra = state.extra

        loss = _to_float(state.last_train_loss)
        lr = _to_float(extra.get("optimizer/lr"))
        attention_lr = _to_float(extra.get("optimizer/attention_lr"))
        grad_norm = _to_float(extra.get("diagnostics/grad_norm"))
        elapsed = _to_float(extra.get("timing/elapsed_since_start_sec"))
        has_nan_or_inf = bool(extra.get("diagnostics/has_nan_or_inf", False))

        parts = [f"{self.prefix} step={state.step}"]

        if loss is not None:
            parts.append(f"loss={loss:.6f}")
        if lr is not None:
            parts.append(f"lr={lr:.6e}")
        if attention_lr is not None:
            parts.append(f"attn_lr={attention_lr:.6e}")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.4f}")
        if elapsed is not None:
            parts.append(f"elapsed={elapsed:.1f}s")
        if has_nan_or_inf:
            parts.append("nan_or_inf=True")

        print(" | ".join(parts))

    def on_eval_end(self, trainer: Any, eval_outputs: Optional[dict[str, Any]] = None) -> None:
        if not getattr(trainer, "is_main", True):
            return
        eval_outputs = eval_outputs or {}
        val_loss = _to_float(eval_outputs.get("val_loss"))
        is_best = bool(eval_outputs.get("is_best", False))

        parts = [f"[eval] step={trainer.state.step}"]
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.6f}")
        parts.append(f"is_best={is_best}")

        print(" | ".join(parts))

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        if not getattr(trainer, "is_main", True):
            return
        print(f"[error] step={trainer.state.step} | {type(exc).__name__}: {exc}")



class WandBCallback(Callback):
    def __init__(
        self,
        project: str,
        name: str | None = None,
        entity: str | None = None,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
        yaml_path: str | None = None,
    ) -> None:
        self.project = project
        self.name = name
        self.entity = entity
        self.run_id = run_id
        self.config = config or {}
        self.tags = tags or []
        self.enabled = enabled
        self.yaml_path = yaml_path

        self._wandb = None
        self._run = None

    def on_run_start(self, trainer: Any) -> None:
        if not self.enabled or not getattr(trainer, "is_main", True):
            return

        import wandb

        # Env vars from TAP (WANDB_PROJECT, WANDB_ENTITY, WANDB_RUN_ID) take
        # priority over config values so TAP can route runs to the right project.
        project = os.environ.get("WANDB_PROJECT") or self.project
        entity  = os.environ.get("WANDB_ENTITY")  or self.entity or None
        run_id  = os.environ.get("WANDB_RUN_ID")  or self.run_id or None

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            name=self.name,
            config=self.config,
            tags=self.tags,
            resume="allow" if run_id else None,
        )
        if self._run is not None:
            _path = f"{self._run.entity}/{self._run.project}/{self._run.id}"
            print(f"TAP_WANDB_RUN_ID={_path}", flush=True)

        if self._run is not None and self.yaml_path is not None:
            import os as _os
            artifact = wandb.Artifact(
                name=_os.path.splitext(_os.path.basename(self.yaml_path))[0],
                type="config",
            )
            artifact.add_file(self.yaml_path)
            self._run.log_artifact(artifact)

        # Write resolved hyperparams to summary immediately after init so that
        # CONFIG_OVERRIDES_JSON values are captured even if the run crashes
        # (high-LR divergence etc.) before on_run_end fires.
        if self._run is not None:
            self._write_hparam_summary(trainer)

    def on_step_end(self, trainer: Any, step_outputs: Optional[dict[str, Any]] = None) -> None:
        if not self.enabled or self._wandb is None or not getattr(trainer, "is_main", True):
            return

        state = trainer.state
        extra = state.extra

        payload: dict[str, Any] = {
            "train/step": state.step,
            "train/epoch": state.epoch,
        }

        if state.last_train_loss is not None:
            payload["primary/train_loss"] = state.last_train_loss

        key_map = {
            "optimizer/lr": "primary/lr",
            "optimizer/attention_lr": "primary/attention_lr",
            "diagnostics/grad_norm": "primary/grad_norm",
            "diagnostics/update_to_param_norm": "primary/update_to_param_norm",
            "diagnostics/final_hidden_l2": "primary/final_hidden_l2",
            "diagnostics/logit_l2": "primary/logit_l2",
            "diagnostics/mean_max_softmax_prob": "primary/mean_max_softmax_prob",
            "diagnostics/has_nan_or_inf_loss": "primary/has_nan_or_inf_loss",
            "diagnostics/has_nan_or_inf_grad": "primary/has_nan_or_inf_grad",
            "diagnostics/has_nan_or_inf": "primary/has_nan_or_inf",
            "timing/elapsed_since_start_sec": "train/elapsed_sec",
            "diagnostics/param_norm": "secondary/param_norm",
            "diagnostics/update_norm": "secondary/update_norm",
            "diagnostics/max_logit": "secondary/max_logit",
            "diagnostics/min_logit": "secondary/min_logit",
            "diagnostics/logsumexp_logits": "secondary/logsumexp_logits",
            "diagnostics/attn_logit_max_layer0": "secondary/attn_logit_max_layer0",
            "diagnostics/attn_logit_max_layer0_log": "secondary/attn_logit_max_layer0_log",
            "sharpness/lambda_max_global": "sharpness/lambda_max_global",
            "sharpness/lambda_max_attention_block": "sharpness/lambda_max_attention_block",
            "sharpness/lambda_max_mlp_block": "sharpness/lambda_max_mlp_block",
            "sharpness/update_norm_times_lambda_max": "sharpness/update_norm_times_lambda_max",
        }

        for src, dst in key_map.items():
            value = extra.get(src)
            if value is not None:
                payload[dst] = value

        for key, value in extra.items():
            if (
                # key.startswith("grad_norm_inspect/") or
                key.startswith("optimizer_inspect/")
                or key.startswith("attention_diagnostics/")
            ) and value is not None:
                payload[key] = value

        self._wandb.log(payload, step=state.step)

    def on_eval_end(self, trainer: Any, eval_outputs: Optional[dict[str, Any]] = None) -> None:
        if not self.enabled or self._wandb is None or not getattr(trainer, "is_main", True):
            return

        state = trainer.state
        eval_outputs = eval_outputs or {}

        payload: dict[str, Any] = {
            "train/step": state.step,
            "train/epoch": state.epoch,
        }

        val_loss = eval_outputs.get("val_loss")
        if _is_number(val_loss):
            payload["primary/val_loss"] = float(val_loss)

        self._wandb.log(payload, step=state.step)

    def _write_hparam_summary(self, trainer: Any) -> None:
        """Write resolved hyperparams to W&B summary.

        Called once at run start so CONFIG_OVERRIDES_JSON values are reflected
        in summary even if the run crashes before on_run_end fires.
        """
        if self._run is None:
            return
        try:
            model = getattr(trainer.model, "module", trainer.model)
            self._run.summary["config/num_layers"] = model.cfg.num_layers
            self._run.summary["config/model_dim"]  = model.cfg.model_dim
            self._run.summary["config/core_params"] = model.count_core_params()
            self._run.summary["config/total_params"] = model.count_params()
        except AttributeError:
            pass

    def on_run_end(self, trainer: Any) -> None:
        if not self.enabled or self._run is None or not getattr(trainer, "is_main", True):
            return

        self._run.finish()
        self._run = None
        self._wandb = None
        
    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        if not self.enabled or self._run is None or not getattr(trainer, "is_main", True):
            return
        self._run.summary["failed"] = True
        self._run.summary["error_type"] = type(exc).__name__
        self._run.summary["error_message"] = str(exc)


##
#attn/layer_1/gap_p95
# attn/layer_1/absmax_p95
# attn/layer_1/maxprob_frac_gt_0.9
# attn/layer_1/entropy_mean
# attn/layer_1/entropy_p05

# attn/layer_4/gap_p95
# attn/layer_4/absmax_p95
# attn/layer_4/maxprob_frac_gt_0.9
# attn/layer_4/entropy_mean
# attn/layer_4/entropy_p05

# attn/layer_8/gap_p95
# attn/layer_8/absmax_p95
# attn/layer_8/maxprob_frac_gt_0.9
# attn/layer_8/entropy_mean
# attn/layer_8/entropy_p05