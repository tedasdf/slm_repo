from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..model.model import TransformerLM

if TYPE_CHECKING:
    from ..model.config import ModelConfig
    from ..training.run_config import TrainerConfig
    from .config import ResourceConfig


_TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class ResourceEstimate:
    """All preflight estimates in one place."""

    # ── Model scale ───────────────────────────────────────────────────────────
    num_params: int | None = None          # total (incl. embedding + head)
    num_core_params: int | None = None     # excl. embedding + head (Kaplan et al.)
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
        batch_size: int = 1,
        world_size: int = 1,
    ) -> None:
        self.model        = model
        self.trainer_cfg  = trainer_cfg
        self.resource_cfg = resource_cfg
        self.batch_size   = batch_size  # per-device micro-batch size (data.batch_size)
        self.world_size   = world_size

    # ── Per-quantity methods ──────────────────────────────────────────────────

    def count_params(self) -> int:
        return self.model.count_params()

    def estimate_param_memory_gb(self) -> float:
        # Reads each parameter's actual dtype, so per-layer dtype overrides
        # (e.g. fp32 norms alongside bf16 linears) are reflected automatically.
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total_bytes / 1024**3

    def estimate_flops_per_step(self) -> float:
        seq_len = self.trainer_cfg.max_seq_len
        tokens_per_step = self.batch_size * self.trainer_cfg.grad_accum_steps * seq_len
        fwd_flops = tokens_per_step * self.model.flops_per_token(seq_len)
        return 3 * fwd_flops  # fwd + bwd (bwd ≈ 2x fwd)

    def estimate_activation_memory_gb(self) -> float:
        cfg = self.model.cfg
        seq_len = self.trainer_cfg.max_seq_len

        # Per-token activations retained for backward, per transformer block:
        #   - residual stream copies before attn-norm and mlp-norm: 2 * model_dim
        #   - q, k, v projections:                                   3 * model_dim
        #   - attention output (pre out-proj):                       1 * model_dim
        #   - mlp hidden activation:                                  hidden_dim
        per_token_elems = 6 * cfg.model_dim + cfg.hidden_dim

        # Flash attention (bf16/fp16, see build_model) never materializes the
        # full [num_heads, seq_len, seq_len] score matrix. The math fallback
        # (fp32) does, so account for it only in that case.
        if self.trainer_cfg.precision == "fp32":
            per_token_elems += cfg.attention.num_heads * seq_len

        # Activations for one microbatch are freed after its backward pass
        # before the next accumulation step starts, so this scales with the
        # real per-device batch_size, not grad_accum_steps.
        activation_elems = cfg.num_layers * self.batch_size * seq_len * per_token_elems
        bytes_per_elem = 4 if self.trainer_cfg.precision == "fp32" else 2
        total_bytes = activation_elems * bytes_per_elem * self.resource_cfg.activation_memory_overhead
        return total_bytes / 1024**3

    def estimate_optimizer_memory_gb(self) -> float:
        # AdamW keeps grad + exp_avg (m) + exp_avg_sq (v) per parameter, each
        # matching that parameter's own dtype/size — same basis as param memory.
        return 3 * self.estimate_param_memory_gb()

    def estimate_step_time_sec(self) -> float:
        # mfu (Model FLOPs Utilization) scales hardware peak FLOPs down to a
        # realistic achieved rate — depends on attention impl, batch/seq size,
        # precision, etc. Treated as a single config constant for now.
        achievable_flops_per_sec = (
            self.resource_cfg.gpu_tflops * 1e12
            * self.resource_cfg.mfu
            * self.world_size
        )
        return self.estimate_flops_per_step() / achievable_flops_per_sec

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def estimate(self) -> ResourceEstimate:
        """Run all estimates; unimplemented methods are silently skipped."""
        est = ResourceEstimate()
        tokens_per_step = self.batch_size * self.trainer_cfg.grad_accum_steps * self.trainer_cfg.max_seq_len

        try:
            est.num_params = self.count_params()
            est.num_core_params = self.model.count_core_params()
        except NotImplementedError:
            pass

        try:
            est.param_mem_gb = self.estimate_param_memory_gb()
        except NotImplementedError:
            pass

        try:
            est.flops_per_step = self.estimate_flops_per_step()
            if est.num_params:
                est.flops_per_token = est.flops_per_step / tokens_per_step
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


def estimate_resources(
    model_cfg: "ModelConfig",
    trainer_cfg: "TrainerConfig",
    resource_cfg: "ResourceConfig",
    *,
    batch_size: int = 1,
    world_size: int = 1,
) -> ResourceEstimate:
    """Build a throwaway model on CPU (cast to trainer_cfg.precision) and run
    the full ResourceEstimator pass."""
    model = TransformerLM(model_cfg)
    model = model.to(_TORCH_DTYPES[trainer_cfg.precision])

    estimator = ResourceEstimator(
        model,
        trainer_cfg,
        resource_cfg,
        batch_size=batch_size,
        world_size=world_size,
    )
    return estimator.estimate()
