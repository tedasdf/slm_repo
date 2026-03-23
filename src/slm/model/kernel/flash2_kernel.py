import math
from typing import Optional

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634

# Optional: print winning configs during autotune
# You can also set this in the shell instead.
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def _require_cuda_fp16_bhtd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, v must be CUDA tensors")
    if not (q.dtype == k.dtype == v.dtype == torch.float16):
        raise ValueError("This minimal implementation expects float16 q/k/v")
    if not (q.ndim == k.ndim == v.ndim == 4):
        raise ValueError("Expected q, k, v to have shape [B, H, T, D]")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q, k, v must have identical shapes")
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("Head dim D must be one of {16, 32, 64, 128}")


# ---------------------------------------------------------------------
# autotune configs
# ---------------------------------------------------------------------

_ATTENTION_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
    for (bm, bn) in (
        (32, 32),
        (64, 32),
        (64, 64),
        (128, 32),
        (128, 64),
        (128, 128),
    )
    for w in (4, 8)
    for s in (2, 3)
]

_DELTA_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=1)
    for bm in (64, 128, 256)
    for w in (4, 8)
]


def _prune_attention_configs(configs, named_args, **kwargs):
    """
    Keep search space reasonable.
    For causal attention, prefer BLOCK_M >= BLOCK_N.
    """
    causal = named_args.get("CAUSAL", kwargs.get("CAUSAL", True))
    if causal:
        configs = [c for c in configs if c.kwargs["BLOCK_M"] >= c.kwargs["BLOCK_N"]]
    return configs or [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2)
    ]


# ---------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------


@triton.autotune(
    configs=_ATTENTION_AUTOTUNE_CONFIGS,
    key=["T", "D", "CAUSAL"],
    prune_configs_by={"early_config_prune": _prune_attention_configs},
    cache_results=True,
)
@triton.jit
def _flash2_fwd_kernel(
    Q,
    K,
    V,
    O,  # noqa: E741
    LSE,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lt,
    H,
    T,
    SM_SCALE,
    SCALE_LOG2,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    row_mask = offs_m < T

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    lse_base = LSE + b * stride_lb + h * stride_lh

    q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

    m_i = tl.where(row_mask, -float("inf"), 0.0)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, T, BLOCK_N):
        cols = start_n + offs_n
        col_mask = cols < T

        k_ptrs = k_base + cols[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = v_base + cols[:, None] * stride_vt + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)

        logits2 = tl.dot(q, tl.trans(k)) * SCALE_LOG2

        valid = row_mask[:, None] & col_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= cols[None, :])

        logits2 = tl.where(valid, logits2, -1.0e6)

        block_max = tl.max(logits2, axis=1)
        new_m = tl.where(row_mask, tl.maximum(m_i, block_max), 0.0)

        p = tl.where(valid, tl.math.exp2(logits2 - new_m[:, None]), 0.0)
        alpha = tl.math.exp2(m_i - new_m)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)

        m_i = new_m

    l_safe = tl.where(row_mask, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse = tl.where(row_mask, m_i + tl.math.log2(l_i), 0.0)

    o_ptrs = o_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.float16), mask=row_mask[:, None])

    lse_ptrs = lse_base + offs_m * stride_lt
    tl.store(lse_ptrs, lse, mask=row_mask)


# ---------------------------------------------------------------------
# backward prepass
# ---------------------------------------------------------------------


@triton.autotune(
    configs=_DELTA_AUTOTUNE_CONFIGS,
    key=["T", "D"],
    cache_results=True,
)
@triton.jit
def _flash2_bwd_delta_kernel(
    O,  # noqa: E741
    DO,
    DELTA,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_db,
    stride_dh,
    stride_dt,
    H,
    T,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_mask = offs_m < T

    o_base = O + b * stride_ob + h * stride_oh
    do_base = DO + b * stride_dob + h * stride_doh
    delta_base = DELTA + b * stride_db + h * stride_dh

    o_ptrs = o_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    do_ptrs = do_base + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod

    o = tl.load(o_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(delta_base + offs_m * stride_dt, delta, mask=row_mask)


# ---------------------------------------------------------------------
# backward dK / dV
# ---------------------------------------------------------------------


@triton.autotune(
    configs=_ATTENTION_AUTOTUNE_CONFIGS,
    key=["T", "D", "CAUSAL"],
    prune_configs_by={"early_config_prune": _prune_attention_configs},
    cache_results=True,
)
@triton.jit
def _flash2_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_dkb,
    stride_dkh,
    stride_dkt,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvt,
    stride_dvd,
    stride_db,
    stride_dh,
    stride_dt,
    H,
    T,
    SM_SCALE,
    SCALE_LOG2,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    col_mask = offs_n < T

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    do_base = DO + b * stride_dob + h * stride_doh
    lse_base = LSE + b * stride_lb + h * stride_lh
    delta_base = DELTA + b * stride_db + h * stride_dh
    dk_base = DK + b * stride_dkb + h * stride_dkh
    dv_base = DV + b * stride_dvb + h * stride_dvh

    k_ptrs = k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

    k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)

    dk_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    for start_m in range(0, T, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        row_mask = offs_m < T

        q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod

        q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0)

        lse = tl.load(lse_base + offs_m * stride_lt, mask=row_mask, other=0.0)
        delta = tl.load(delta_base + offs_m * stride_dt, mask=row_mask, other=0.0)

        logits2_t = tl.dot(k, tl.trans(q)) * SCALE_LOG2

        valid = col_mask[:, None] & row_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[None, :] >= offs_n[:, None])

        p_t = tl.where(valid, tl.math.exp2(logits2_t - lse[None, :]), 0.0)

        dv_acc += tl.dot(p_t.to(tl.float16), do)
        dp_t = tl.dot(v, tl.trans(do)).to(tl.float32)

        ds_t = tl.where(valid, p_t * (dp_t - delta[None, :]), 0.0)
        dk_acc += tl.dot(ds_t.to(tl.float16), q)

    dk_acc *= SM_SCALE

    dk_ptrs = dk_base + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = dv_base + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk_acc.to(tl.float16), mask=col_mask[:, None])
    tl.store(dv_ptrs, dv_acc.to(tl.float16), mask=col_mask[:, None])


# ---------------------------------------------------------------------
# backward dQ
# ---------------------------------------------------------------------


@triton.autotune(
    configs=_ATTENTION_AUTOTUNE_CONFIGS,
    key=["T", "D", "CAUSAL"],
    prune_configs_by={"early_config_prune": _prune_attention_configs},
    cache_results=True,
)
@triton.jit
def _flash2_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_dqb,
    stride_dqh,
    stride_dqt,
    stride_dqd,
    stride_db,
    stride_dh,
    stride_dt,
    H,
    T,
    SM_SCALE,
    SCALE_LOG2,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_mask = offs_m < T

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    do_base = DO + b * stride_dob + h * stride_doh
    lse_base = LSE + b * stride_lb + h * stride_lh
    delta_base = DELTA + b * stride_db + h * stride_dh
    dq_base = DQ + b * stride_dqb + h * stride_dqh

    q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    do_ptrs = do_base + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod

    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0)

    lse = tl.load(lse_base + offs_m * stride_lt, mask=row_mask, other=0.0)
    delta = tl.load(delta_base + offs_m * stride_dt, mask=row_mask, other=0.0)

    dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, T, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        col_mask = offs_n < T

        k_ptrs = k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)

        logits2 = tl.dot(q, tl.trans(k)) * SCALE_LOG2

        valid = row_mask[:, None] & col_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        p = tl.where(valid, tl.math.exp2(logits2 - lse[:, None]), 0.0)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        dq_acc += tl.dot(ds.to(tl.float16), k)

    dq_acc *= SM_SCALE

    dq_ptrs = dq_base + offs_m[:, None] * stride_dqt + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(tl.float16), mask=row_mask[:, None])


class _FlashAttn2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        sm_scale: float,
    ) -> torch.Tensor:
        _require_cuda_fp16_bhtd(q, k, v)

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        B, H, T, D = q.shape
        o = torch.empty_like(q)
        lse = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)  # noqa: E731

        _flash2_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            H,
            T,
            SM_SCALE=sm_scale,
            SCALE_LOG2=sm_scale * LOG2E,
            D=D,
            CAUSAL=causal,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, o, lse = ctx.saved_tensors
        causal = ctx.causal
        sm_scale = ctx.sm_scale

        if not do.is_contiguous():
            do = do.contiguous()

        B, H, T, D = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

        pre_grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)  # noqa: E731
        _flash2_bwd_delta_kernel[pre_grid](
            o,
            do,
            delta,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            H,
            T,
            D=D,
        )

        grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * H)  # noqa: E731
        _flash2_bwd_dkdv_kernel[grid_kv](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            H,
            T,
            SM_SCALE=sm_scale,
            SCALE_LOG2=sm_scale * LOG2E,
            D=D,
            CAUSAL=causal,
        )

        grid_q = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)  # noqa: E731
        _flash2_bwd_dq_kernel[grid_q](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            H,
            T,
            SM_SCALE=sm_scale,
            SCALE_LOG2=sm_scale * LOG2E,
            D=D,
            CAUSAL=causal,
        )

        return dq, dk, dv, None, None


def flash_attn2_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    _require_cuda_fp16_bhtd(q, k, v)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return _FlashAttn2Triton.apply(q, k, v, causal, sm_scale)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * sm_scale

    if causal:
        T = q.shape[-2]
        mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, -float("inf"))

    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


# ---------------------------------------------------------------------
# benchmarking
# ---------------------------------------------------------------------


def _attention_tflops(
    B: int, H: int, T: int, D: int, causal: bool, mode: str, ms: float
) -> float:
    flops_per_matmul = 2.0 * B * H * T * T * D
    total_flops = 2.0 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5
    return total_flops * 1e-12 / (ms * 1e-3)


def benchmark_flash2_once(
    B: int = 2,
    H: int = 4,
    T: int = 512,
    D: int = 64,
    causal: bool = True,
    mode: str = "fwd",
    warmup: int = 25,
    rep: int = 100,
):
    device = "cuda"
    dtype = torch.float16

    q = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((B, H, T, D), device=device, dtype=dtype, requires_grad=True)

    sm_scale = 1.0 / math.sqrt(D)

    if mode == "fwd":
        fn = lambda: flash_attn2_triton(q, k, v, causal=causal, sm_scale=sm_scale)  # noqa: E731
    elif mode == "bwd":
        o = flash_attn2_triton(q, k, v, causal=causal, sm_scale=sm_scale)
        do = torch.randn_like(o)

        def fn():
            q.grad = None
            k.grad = None
            v.grad = None
            o.backward(do, retain_graph=True)
    else:
        raise ValueError("mode must be 'fwd' or 'bwd'")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    tflops = _attention_tflops(B, H, T, D, causal, mode, ms)

    print(
        f"[{mode}] B={B} H={H} T={T} D={D} causal={causal} -> {ms:.3f} ms, {tflops:.2f} TFLOPS"
    )
    return ms, tflops


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg="mode",
        line_vals=["fwd", "bwd"],
        line_names=["forward", "backward"],
        plot_name="flash2_triton_autotuned",
        args={"B": 2, "H": 4, "D": 64, "causal": True},
        xlabel="sequence length",
        ylabel="TFLOPS",
        x_log=True,
    )
)
def bench_flash2(B, H, T, D, causal, mode):
    _, tflops = benchmark_flash2_once(B=B, H=H, T=T, D=D, causal=causal, mode=mode)
    return tflops


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    B, H, T, D = 2, 4, 512, 64

    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    # first call will autotune for this (T, D, causal) shape
    out = flash_attn2_triton(q, k, v, causal=True)
    out_ref = reference_attention(q_ref, k_ref, v_ref, causal=True)

    loss = out.float().square().mean()
    loss_ref = out_ref.float().square().mean()

    loss.backward()
    loss_ref.backward()

    print("forward max abs err:", (out - out_ref).abs().max().item())
    print("dq max abs err     :", (q.grad - q_ref.grad).abs().max().item())
    print("dk max abs err     :", (k.grad - k_ref.grad).abs().max().item())
    print("dv max abs err     :", (v.grad - v_ref.grad).abs().max().item())

    # one-off timings
    benchmark_flash2_once(B=2, H=4, T=512, D=64, causal=True, mode="fwd")
    benchmark_flash2_once(B=2, H=4, T=512, D=64, causal=True, mode="bwd")

    # sweep and save plot / data to current directory
    bench_flash2.run(save_path=".", print_data=True)
