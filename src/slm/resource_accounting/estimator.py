from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model.model import TransformerLM
    from ..training.run_config import TrainerConfig
    from .config import ResourceConfig


@dataclass
class ResourceEstimate:
    """All preflight estimates in one place."""

    # ── Model scale ───────────────────────────────────────────────────────────
    num_params: int | None = None
    param_mem_gb: float | None = None

    # ── Per-step compute ──────────────────────────────────────────────────────
    flops_per_step: float | None = None
    flops_per_token: float | None = None

    # ── Memory breakdown ──────────────────────────────────────────────────────
    activation_mem_gb: float | None = None
    optimizer_mem_gb: float | None = None
    total_mem_gb: float | None = None

    # ── Time projection ───────────────────────────────────────────────────────
    est_step_time_sec: float | None = None
    est_total_hours: float | None = None


class ResourceEstimator:
    """Top-level resource estimator.

    Delegates count_params / flops_per_token to the model's own methods
    (each nn.Module subclass defines these), then adds training-specific
    estimates (memory, optimizer state, wall time).

    Usage:
        model = TransformerLM(model_cfg)   # CPU is fine — no GPU memory used
        est = ResourceEstimator(model, trainer_cfg, resource_cfg)
        result = est.estimate()

        # inspect individual components directly:
        model.blocks[0].attn.count_params()
        model.blocks[0].mlp.flops_per_token()
    """

    def __init__(
        self,
        model: "TransformerLM",
        trainer_cfg: "TrainerConfig",
        resource_cfg: "ResourceConfig",
        *,
        world_size: int = 1,
    ) -> None:
        self.model        = model
        self.trainer_cfg  = trainer_cfg
        self.resource_cfg = resource_cfg
        self.world_size   = world_size

    # ── Per-quantity methods ──────────────────────────────────────────────────

    def count_params(self) -> int:
        return self.model.count_params()

    def estimate_param_memory_gb(self) -> float:
        # TODO: implement — self.count_params() and self.trainer_cfg.precision
        #   fp32 → 4 bytes/param,  bf16/fp16 → 2 bytes/param
        raise NotImplementedError

    def estimate_flops_per_step(self) -> float:
        # TODO: implement
        #   tokens_per_step = trainer_cfg.grad_accum_steps * trainer_cfg.max_seq_len
        #   fwd_flops = tokens_per_step * self.model.flops_per_token(trainer_cfg.max_seq_len)
        #   total ≈ 3 * fwd_flops  (bwd ≈ 2× fwd)
        raise NotImplementedError

    def estimate_activation_memory_gb(self) -> float:
        # TODO: implement
        #   self.model.cfg, self.trainer_cfg, self.resource_cfg.activation_memory_overhead
        raise NotImplementedError

    def estimate_optimizer_memory_gb(self) -> float:
        # TODO: implement — AdamW keeps 3 fp32 copies (param + m + v)
        #   self.estimate_param_memory_gb(), self.trainer_cfg.precision
        raise NotImplementedError

    def estimate_step_time_sec(self) -> float:
        # TODO: implement
        #   self.estimate_flops_per_step() / (resource_cfg.gpu_tflops * 1e12 * resource_cfg.mfu * world_size)
        raise NotImplementedError

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def estimate(self) -> ResourceEstimate:
        """Run all estimates; unimplemented methods are silently skipped."""
        est = ResourceEstimate()
        batch_size = self.trainer_cfg.grad_accum_steps
        seq_len    = self.trainer_cfg.max_seq_len

        try:
            est.num_params = self.count_params()
        except NotImplementedError:
            pass

        try:
            est.param_mem_gb = self.estimate_param_memory_gb()
        except NotImplementedError:
            pass

        try:
            est.flops_per_step = self.estimate_flops_per_step()
            if est.num_params:
                est.flops_per_token = est.flops_per_step / (batch_size * seq_len)
        except NotImplementedError:
            pass

        try:
            est.activation_mem_gb = self.estimate_activation_memory_gb()
        except NotImplementedError:
            pass

        try:
            est.optimizer_mem_gb = self.estimate_optimizer_memory_gb()
        except NotImplementedError:
            pass

        parts = [est.param_mem_gb, est.activation_mem_gb, est.optimizer_mem_gb]
        if any(p is not None for p in parts):
            est.total_mem_gb = sum(p for p in parts if p is not None)

        try:
            est.est_step_time_sec = self.estimate_step_time_sec()
            if est.est_step_time_sec is not None:
                est.est_total_hours = (
                    est.est_step_time_sec * self.trainer_cfg.max_steps / 3600.0
                )
        except NotImplementedError:
            pass

        return est
