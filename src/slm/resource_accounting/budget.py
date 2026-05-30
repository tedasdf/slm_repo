from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ResourceConfig
    from .estimator import ResourceEstimate


class FitStatus(str, Enum):
    OK      = "ok"       # comfortably within budget
    TIGHT   = "tight"    # within budget but < 10 % headroom
    OOM     = "oom"      # predicted out-of-memory
    UNKNOWN = "unknown"  # couldn't estimate (equations not yet implemented)


@dataclass
class BudgetReport:
    status: FitStatus

    # Total memory budget available across all GPUs
    budget_gb: float

    # Estimated memory required (from ResourceEstimate.total_mem_gb)
    required_gb: float | None

    # How much headroom is left as a fraction of budget (None if unknown)
    headroom_fraction: float | None

    # Human-readable one-line summary
    summary: str

    def is_safe(self) -> bool:
        """True if training can proceed (ok or tight)."""
        return self.status in (FitStatus.OK, FitStatus.TIGHT)


def check_budget(
    estimate: "ResourceEstimate",
    resource_cfg: "ResourceConfig",
    *,
    world_size: int = 1,
) -> BudgetReport:
    """Compare the memory estimate against the available GPU budget.

    Budget = gpu_mem_gb * world_size * memory_budget_fraction

    Returns a BudgetReport with a FitStatus and a plain-English summary.
    Callers (callback, reporter) use report.is_safe() / report.status to
    decide whether to warn, abort, or proceed silently.
    """
    total_budget_gb = (
        resource_cfg.gpu_mem_gb
        * world_size
        * resource_cfg.memory_budget_fraction
    )

    required = estimate.total_mem_gb

    if required is None:
        return BudgetReport(
            status=FitStatus.UNKNOWN,
            budget_gb=total_budget_gb,
            required_gb=None,
            headroom_fraction=None,
            summary=(
                "Memory fit-check skipped — estimation equations not yet implemented. "
                f"Available budget: {total_budget_gb:.1f} GB across {world_size} GPU(s)."
            ),
        )

    headroom = (total_budget_gb - required) / total_budget_gb

    if required > total_budget_gb:
        status = FitStatus.OOM
        summary = (
            f"OOM predicted: need ~{required:.1f} GB but budget is "
            f"{total_budget_gb:.1f} GB ({world_size}× GPU). "
            "Consider reducing batch size, seq_len, or model size."
        )
    elif headroom < 0.10:
        status = FitStatus.TIGHT
        summary = (
            f"Tight fit: ~{required:.1f} GB required vs {total_budget_gb:.1f} GB budget "
            f"({headroom*100:.1f} % headroom). Watch for fragmentation OOMs."
        )
    else:
        status = FitStatus.OK
        summary = (
            f"Fits comfortably: ~{required:.1f} GB required vs "
            f"{total_budget_gb:.1f} GB budget ({headroom*100:.1f} % headroom)."
        )

    return BudgetReport(
        status=status,
        budget_gb=total_budget_gb,
        required_gb=required,
        headroom_fraction=headroom,
        summary=summary,
    )