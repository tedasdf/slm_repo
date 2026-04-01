from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0

    train_tokens_seen: int = 0
    train_samples_seen: int = 0

    best_val_loss: Optional[float] = None
    last_train_loss: Optional[float] = None
    last_val_loss: Optional[float] = None

    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    should_stop: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_best_val(self) -> bool:
        return self.best_val_loss is not None

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self.started_at is None:
            return None
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    def update_best_val(self, val_loss: float) -> bool:
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False