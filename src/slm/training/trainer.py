from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from .callbacks import CallbackList
from .run_config import TrainerConfig
from .state import TrainState


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)
    return batch


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Any,
        config: TrainerConfig,
        *,
        val_loader: Any | None = None,
        scheduler: Any | None = None,
        callbacks: list[Any] | None = None,
        loss_fn: Any | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.loss_fn = loss_fn

        self.callbacks = CallbackList(callbacks or [])
        self.state = TrainState()

        self.device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        print(self.device)
        raise ValueError


        if self.config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self._use_grad_scaler())

    @classmethod
    def from_components(cls, components: dict[str, Any]) -> "Trainer":
        required = [
            "model",
            "optimizer",
            "scheduler",
            "train_loader",
            "val_loader",
            "trainer_cfg",
            "callbacks",
        ]
        missing = [k for k in required if k not in components]
        if missing:
            raise KeyError(f"Missing trainer components: {missing}")

        return cls(
            model=components["model"],
            optimizer=components["optimizer"],
            scheduler=components["scheduler"],
            train_loader=components["train_loader"],
            val_loader=components["val_loader"],
            config=components["trainer_cfg"],
            callbacks=components["callbacks"],
        )

    def _use_autocast(self) -> bool:
        return self.device.type == "cuda" and self.config.precision in {"fp16", "bf16"}

    def _use_grad_scaler(self) -> bool:
        return self.device.type == "cuda" and self.config.precision == "fp16"

    def _autocast_dtype(self) -> torch.dtype:
        if self.config.precision == "fp16":
            return torch.float16
        if self.config.precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def _autocast_context(self):
        if not self._use_autocast():
            return contextlib.nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype(),
        )

    def _extract_model_inputs(self, batch: Any) -> tuple[Any, Any | None]:
        if isinstance(batch, dict):
            if "input_ids" in batch and "targets" in batch:
                return {"input_ids": batch["input_ids"]}, batch["targets"]
            if "input_ids" in batch and "labels" in batch:
                return {"input_ids": batch["input_ids"]}, batch["labels"]
            return batch, None

        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                return batch[0], batch[1]
            if len(batch) == 1:
                return batch[0], None

        return batch, None

    def _compute_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        batch = move_to_device(batch, self.device)
        model_inputs, targets = self._extract_model_inputs(batch)

        with self._autocast_context():
            if isinstance(model_inputs, dict):
                outputs = self.model(**model_inputs, targets=targets)
            else:
                outputs = self.model(model_inputs, targets=targets)

            if isinstance(outputs, dict):
                if "loss" in outputs:
                    loss = outputs["loss"]
                elif self.loss_fn is not None:
                    logits = outputs["logits"]
                    if targets is None:
                        raise ValueError("targets are required when using loss_fn")
                    loss = self.loss_fn(logits, targets)
                else:
                    raise ValueError("Model outputs dict without 'loss' and no loss_fn was provided")
            else:
                if self.loss_fn is None:
                    raise ValueError("Non-dict model outputs require an explicit loss_fn")
                if targets is None:
                    raise ValueError("targets are required when using loss_fn")
                loss = self.loss_fn(outputs, targets)

        if not torch.isfinite(loss):
            self.state.extra["diagnostics/has_nan_or_inf_loss"] = True
        else:
            self.state.extra["diagnostics/has_nan_or_inf_loss"] = False

        return loss, outputs if isinstance(outputs, dict) else {"outputs": outputs}

    def _compute_grad_norm(self) -> float | None:
        total_sq = 0.0
        found = False
        for p in self.model.parameters():
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            total_sq += grad.norm(2).item() ** 2
            found = True
        if not found:
            return None
        return math.sqrt(total_sq)

    def train_step(self, batch: Any) -> dict[str, Any]:
        step_start = time.time()
        self.callbacks.on_step_start(self)

        loss, outputs = self._compute_loss(batch)
        loss_for_backward = loss / self.config.grad_accum_steps

        if self._use_grad_scaler():
            self.grad_scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        self.state.last_train_loss = float(loss.detach().item())

        step_outputs: dict[str, Any] = {
            "loss": self.state.last_train_loss,
        }

        return step_outputs

    def optimizer_step(self) -> dict[str, Any]:
        grad_norm = None

        if self.config.clip_grad_norm is not None:
            if self._use_grad_scaler():
                self.grad_scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm,
                ).item()
            )
        else:
            grad_norm = self._compute_grad_norm()

        self.state.extra["diagnostics/grad_norm"] = grad_norm

        if self._use_grad_scaler():
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.param_groups[0]["lr"]
        self.state.extra["optimizer/lr"] = lr

        return {
            "grad_norm": grad_norm,
            "lr": lr,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.callbacks.on_eval_start(self)

        total_loss = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(self.val_loader):
            if self.config.max_eval_batches is not None and batch_idx >= self.config.max_eval_batches:
                break

            loss, _ = self._compute_loss(batch)
            total_loss += float(loss.detach().item())
            total_batches += 1

        val_loss = total_loss / max(total_batches, 1)
        self.state.last_val_loss = val_loss
        is_best = self.state.update_best_val(val_loss)

        eval_outputs = {
            "val_loss": val_loss,
            "is_best": is_best,
            "num_eval_batches": total_batches,
        }

        self.callbacks.on_eval_end(self, eval_outputs)
        self.model.train()

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "state": {
                "step": self.state.step,
                "epoch": self.state.epoch,
                "train_tokens_seen": self.state.train_tokens_seen,
                "train_samples_seen": self.state.train_samples_seen,
                "best_val_loss": self.state.best_val_loss,
                "last_train_loss": self.state.last_train_loss,
                "last_val_loss": self.state.last_val_loss,
                "extra": self.state.extra,
            },
            "trainer_config": self.config.__dict__,
        }
        torch.save(ckpt, path)
        self.callbacks.on_checkpoint_save(self, str(path))

    def train(self) -> TrainState:
        self.state.started_at = time.time()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if self.config.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)

        self.callbacks.on_run_start(self)
        self.callbacks.on_train_start(self)

        try:
            if self.config.num_sanity_val_steps > 0 and self.val_loader is not None:
                self.validate()

            while self.state.step < self.config.max_steps and not self.state.should_stop:
                self.state.epoch += 1
                self.callbacks.on_epoch_start(self)

                for batch_idx, batch in enumerate(self.train_loader):
                    if self.state.step >= self.config.max_steps or self.state.should_stop:
                        break

                    step_outputs = self.train_step(batch)

                    if ((batch_idx + 1) % self.config.grad_accum_steps) == 0:
                        opt_outputs = self.optimizer_step()
                        step_outputs.update(opt_outputs)

                        self.state.step += 1

                        step_time = time.time() - self.state.started_at if self.state.started_at else None



                        self.state.extra["timing/elapsed_since_start_sec"] = step_time

                        if self.state.step % self.config.train_log_every == 0:
                            self.callbacks.on_step_end(self, step_outputs)

                        if self.val_loader is not None and self.state.step % self.config.eval_every == 0:
                            self.validate()
    

                        if self.state.step % self.config.checkpoint_every == 0:
                            ckpt_path = Path("artifacts/checkpoints") / f"step_{self.state.step}.pt"
                            self.save_checkpoint(ckpt_path)

                self.callbacks.on_epoch_end(self)

            self.callbacks.on_train_end(self)

        except BaseException as exc:
            self.callbacks.on_exception(self, exc)
            raise
        finally:
            self.state.ended_at = time.time()
            self.callbacks.on_run_end(self)

        return self.state