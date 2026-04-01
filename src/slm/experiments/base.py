from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from slm.training.run_config import RunConfig


class BaseExperiment(ABC):
    def __init__(self, base_cfg: RunConfig) -> None:
        self.base_cfg = base_cfg

    @abstractmethod
    def make_sweep_config(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def apply_overrides(self, cfg: RunConfig, sweep_cfg: Any) -> RunConfig:
        raise NotImplementedError

    @abstractmethod
    def train_one_run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, count: int | None = None) -> str:
        raise NotImplementedError

    def analyze_results(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented analyze_results() yet."
        )