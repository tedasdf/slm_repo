from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .distributed import all_reduce_sum, barrier

from .callbacks import CallbackList
from .run_config import TrainerConfig
from .state import TrainState

from ..data.tokenizer import BPETokenizer
from ..data.tokenization import maybe_tokenize_batch, fit_or_load_tokenizer_from_loader

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
        tokenizer: BPETokenizer | None = None,
        tokenizer_cfg: Any | None = None,
        val_loader: Any | None = None,
        scheduler: Any | None = None,
        callbacks: list[Any] | None = None,
        loss_fn: Any | None = None,
        dist_env: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_cfg = tokenizer_cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.loss_fn = loss_fn

        self.callbacks = CallbackList(callbacks or [])
        self.state = TrainState()

        if dist_env is None:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.is_distributed = False
            self.is_main = True
            self.device = torch.device(
                "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
            )
        else:
            self.rank = dist_env.rank
            self.local_rank = dist_env.local_rank
            self.world_size = dist_env.world_size
            self.is_distributed = dist_env.is_distributed
            self.is_main = dist_env.is_main
            self.device = dist_env.device

        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self._use_grad_scaler(),
        )

        self.state.extra["distributed/rank"] = self.rank
        self.state.extra["distributed/local_rank"] = self.local_rank
        self.state.extra["distributed/world_size"] = self.world_size
        
    @classmethod
    def from_components(
        cls,
        components: dict[str, Any],
        *,
        dist_env: Any | None = None,
    ) -> "Trainer":
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
            tokenizer=components.get("tokenizer"),
            tokenizer_cfg=components.get("tokenizer_cfg"),
            dist_env=dist_env,
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

    def _tokenize_batch_if_needed(self, batch: Any) -> Any:
        out = maybe_tokenize_batch(
            batch,
            self.tokenizer,
            text_key=getattr(self.config, "text_key", "text"),
            eos_token=(
                getattr(self.tokenizer_cfg, "eos_token", None)
                if self.tokenizer_cfg is not None
                else None
            ),
            pad_token=(
                getattr(self.tokenizer_cfg, "pad_token", None)
                if self.tokenizer_cfg is not None
                else None
            ),
            append_eos=(
                getattr(self.tokenizer_cfg, "append_eos", True)
                if self.tokenizer_cfg is not None
                else True
            ),
            max_seq_len=getattr(self.config, "max_seq_len", None),
        )

        if (
            out is batch
            and isinstance(batch, dict)
            and getattr(self.config, "text_key", "text") in batch
        ):
            raise ValueError(
                "Received raw-text batch but no tokenizer is available. "
                "Provide a tokenizer or enable tokenizer fitting before training."
            )

        return out

    def _compute_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any], Any | None]:
        batch = self._tokenize_batch_if_needed(batch)
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

        self.state.extra["diagnostics/has_nan_or_inf_loss"] = not torch.isfinite(loss)

        return loss, outputs if isinstance(outputs, dict) else {"outputs": outputs}, targets

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

    def _fit_tokenizer_if_needed(self) -> None:
        if self.tokenizer is not None:
            return

        if not getattr(self.config, "train_tokenizer_before_fit", False):
            return

        if self.tokenizer_cfg is None:
            raise ValueError(
                "train_tokenizer_before_fit=True but tokenizer_cfg was not provided."
            )

        common_kwargs = dict(
            vocab_size=self.tokenizer_cfg.vocab_size,
            special_tokens=self.tokenizer_cfg.special_tokens,
            unk_token=self.tokenizer_cfg.unk_token,
            min_frequency=self.tokenizer_cfg.min_frequency,
            text_key=getattr(self.config, "text_key", "text"),
            max_train_texts=getattr(self.tokenizer_cfg, "tokenizer_train_samples", None),
            tokenizer_path=getattr(self.tokenizer_cfg, "tokenizer_path", None),
            reuse_existing=getattr(self.tokenizer_cfg, "reuse_existing", True),
        )

        if self.is_distributed and not self.is_main:
            barrier()
            self.tokenizer = fit_or_load_tokenizer_from_loader(
                self.train_loader,
                **common_kwargs,
                save_if_trained=False,
            )
            return

        self.tokenizer = fit_or_load_tokenizer_from_loader(
            self.train_loader,
            **common_kwargs,
            save_if_trained=True,
        )

        if self.is_distributed:
            barrier()


    def train_step(self, batch: Any) -> dict[str, Any]:
        self.callbacks.on_step_start(self)

        loss, outputs, targets = self._compute_loss(batch)
        loss_for_backward = loss / self.config.grad_accum_steps

        if self._use_grad_scaler():
            self.grad_scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        self.state.last_train_loss = float(loss.detach().item())

        tokens_this_batch = 0
        samples_this_batch = 0

        if torch.is_tensor(targets):
            local_tokens = int((targets != -100).sum().item())
            local_batch_size = int(targets.size(0)) if targets.ndim > 0 else 0

            tokens_this_batch = int(all_reduce_sum(local_tokens, self.device))
            samples_this_batch = int(all_reduce_sum(local_batch_size, self.device))

            self.state.train_tokens_seen += tokens_this_batch
            self.state.train_samples_seen += samples_this_batch

        step_outputs: dict[str, Any] = {
            "loss": self.state.last_train_loss,
            "tokens_this_batch": tokens_this_batch,
            "samples_this_batch": samples_this_batch,
            "train_tokens_seen": self.state.train_tokens_seen,
            "train_samples_seen": self.state.train_samples_seen,
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

        local_loss_sum = 0.0
        local_batches = 0

        try:
            for batch_idx, batch in enumerate(self.val_loader):
                if (
                    self.config.max_eval_batches is not None
                    and batch_idx >= self.config.max_eval_batches
                ):
                    break

                loss, _, _ = self._compute_loss(batch)
                local_loss_sum += float(loss.detach().item())
                local_batches += 1

            total_loss_sum = float(all_reduce_sum(local_loss_sum, self.device))
            total_batches = int(all_reduce_sum(local_batches, self.device))

            val_loss = total_loss_sum / max(total_batches, 1)
            self.state.last_val_loss = val_loss
            is_best = self.state.update_best_val(val_loss)

            eval_outputs = {
                "val_loss": val_loss,
                "is_best": is_best,
                "num_eval_batches": total_batches,
            }

            self.callbacks.on_eval_end(self, eval_outputs)
            return eval_outputs

        finally:
            self.model.train()
    
    def save_checkpoint(self, path: str | Path) -> None:
        if not self.is_main:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        ckpt = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
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
        is_best = (
            self.state.last_val_loss is not None
            and self.state.last_val_loss == self.state.best_val_loss
        )
        self.callbacks.on_checkpoint_save(self, str(path), is_best=is_best)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu")

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        saved = ckpt.get("state", {})
        self.state.step = saved.get("step", 0)
        self.state.epoch = saved.get("epoch", 0)
        self.state.train_tokens_seen = saved.get("train_tokens_seen", 0)
        self.state.train_samples_seen = saved.get("train_samples_seen", 0)
        self.state.best_val_loss = saved.get("best_val_loss")
        self.state.last_train_loss = saved.get("last_train_loss")
        self.state.last_val_loss = saved.get("last_val_loss")

        print(f"Resumed from step {self.state.step}, best_val_loss={self.state.best_val_loss}")

    def train(self) -> TrainState:
        self.state.started_at = time.time()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if self.config.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)

        self.callbacks.on_run_start(self)
        self.callbacks.on_train_start(self)

        try:
            self._fit_tokenizer_if_needed()

            if self.config.num_sanity_val_steps > 0 and self.val_loader is not None:
                eval_outputs = self.validate()
                eval_outputs["step"] = self.state.step
                eval_outputs["sanity_check"] = True
                self.callbacks.on_step_end(self, eval_outputs)

            while self.state.step < self.config.max_steps and not self.state.should_stop:
                self.state.epoch += 1
                self.callbacks.on_epoch_start(self)

                sampler = getattr(self.train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.state.epoch)

                for batch_idx, batch in enumerate(self.train_loader):
                    if self.state.step >= self.config.max_steps or self.state.should_stop:
                        break

                    step_outputs = self.train_step(batch)

                    if ((batch_idx + 1) % self.config.grad_accum_steps) == 0:
                        opt_outputs = self.optimizer_step()
                        step_outputs.update(opt_outputs)

                        self.state.step += 1

                        step_time = (
                            time.time() - self.state.started_at
                            if self.state.started_at
                            else None
                        )
                        self.state.extra["timing/elapsed_since_start_sec"] = step_time

                        if step_time and step_time > 0:
                            self.state.extra["throughput/tokens_per_sec"] = (
                                self.state.train_tokens_seen / step_time
                            )

                        if self.device.type == "cuda":
                            self.state.extra["gpu/memory_reserved_gb"] = (
                                torch.cuda.memory_reserved(self.device) / 1024 ** 3
                            )

                        target_train_tokens = getattr(self.config, "target_train_tokens", None)
                        if (
                            target_train_tokens is not None
                            and self.state.train_tokens_seen >= int(target_train_tokens)
                        ):
                            self.state.should_stop = True
                            self.state.extra["stopping/reached_target_train_tokens"] = True

                        if self.state.step % self.config.train_log_every == 0:
                            self.callbacks.on_step_end(self, step_outputs)

                        if self.val_loader is not None and self.state.step % self.config.eval_every == 0:
                            eval_outputs = self.validate()
                            self.callbacks.on_step_end(self, eval_outputs)

                        if (
                            self.config.save_checkpoints is True
                            and self.state.step % self.config.checkpoint_every == 0
                        ):
                            ckpt_path = Path(self.config.checkpoint_dir) / f"step_{self.state.step}.pt"
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