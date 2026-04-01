from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from slm.training.callbacks import Callback


@dataclass
class ExperimentArtifacts:
    train_loader: Any | None = None
    val_loader: Any | None = None
    test_loader: Any | None = None
    test_cases: dict[str, Any] = field(default_factory=dict)
    extra_state: dict[str, Any] = field(default_factory=dict)


class BaseExperiment(ABC):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def build_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self, model: Any) -> Any:
        raise NotImplementedError

    def build_scheduler(self, optimizer: Any) -> Any | None:
        return None

    @abstractmethod
    def build_dataloaders(self) -> ExperimentArtifacts:
        raise NotImplementedError

    def generate_test_cases(self) -> dict[str, Any]:
        return {}

    def build_callbacks(self) -> list[Callback]:
        return []

    def build_metrics(self) -> dict[str, Any]:
        return {}

    def build_loss_fn(self) -> Any | None:
        return None

    def post_eval(self, trainer: Any, eval_outputs: dict[str, Any]) -> None:
        pass

    def post_train(self, trainer: Any) -> None:
        pass