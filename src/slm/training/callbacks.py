from __future__ import annotations

from typing import Any, Optional


class Callback:
    def on_run_start(self, trainer: Any) -> None:
        pass

    def on_run_end(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, trainer: Any) -> None:
        pass

    def on_epoch_end(self, trainer: Any) -> None:
        pass

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_step_start(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, step_outputs: Optional[dict[str, Any]] = None) -> None:
        pass

    def on_eval_start(self, trainer: Any) -> None:
        pass

    def on_eval_end(self, trainer: Any, eval_outputs: Optional[dict[str, Any]] = None) -> None:
        pass

    def on_checkpoint_save(self, trainer: Any, checkpoint_path: str) -> None:
        pass

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        pass


class CallbackList:
    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_run_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_run_start(trainer)

    def on_run_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_run_end(trainer)

    def on_epoch_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(trainer)

    def on_epoch_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer)

    def on_train_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_step_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_step_start(trainer)

    def on_step_end(self, trainer: Any, step_outputs: Optional[dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_step_end(trainer, step_outputs)

    def on_eval_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_eval_start(trainer)

    def on_eval_end(self, trainer: Any, eval_outputs: Optional[dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_eval_end(trainer, eval_outputs)

    def on_checkpoint_save(self, trainer: Any, checkpoint_path: str) -> None:
        for cb in self.callbacks:
            cb.on_checkpoint_save(trainer, checkpoint_path)

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        for cb in self.callbacks:
            cb.on_exception(trainer, exc)