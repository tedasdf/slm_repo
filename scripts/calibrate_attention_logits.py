from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from src.slm.data.tokenization import maybe_tokenize_batch
from src.slm.main import load_config
from src.slm.model import TransformerLM
from src.slm.model.rope import apply_rope, build_rope_cache
from src.slm.training.builders import assemble_training_components
from src.slm.training.trainer import move_to_device
from src.slm.utils.seed import seed_everything


@dataclass
class CapturedScores:
    scores: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _quantile(x: torch.Tensor, q: float) -> float:
    x = x.detach().flatten()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float(), q).item())


def _make_synthetic_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        generator=generator,
        dtype=torch.long,
    )
    return {"input_ids": input_ids.to(device)}


def _prepare_dataloader_batch(
    *,
    cfg: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    components = assemble_training_components(
        cfg,
        rank=0,
        world_size=1,
        is_distributed=False,
        is_main=False,
    )
    batch = next(iter(components["train_loader"]))
    batch = maybe_tokenize_batch(
        batch,
        components["tokenizer"],
        text_key=getattr(cfg.trainer, "text_key", "text"),
        eos_token=getattr(cfg.tokenizer, "eos_token", None),
        pad_token=getattr(cfg.tokenizer, "pad_token", None),
        append_eos=getattr(cfg.tokenizer, "append_eos", True),
        max_seq_len=getattr(cfg.trainer, "max_seq_len", None),
    )
    batch = move_to_device(batch, device)
    if not isinstance(batch, dict) or "input_ids" not in batch:
        raise ValueError("Expected a tokenized batch containing input_ids.")
    return {"input_ids": batch["input_ids"]}


def _capture_base_scores(
    *,
    model: TransformerLM,
    input_ids: torch.Tensor,
    layers: Iterable[int],
) -> dict[int, CapturedScores]:
    wanted = set(layers)
    captures: dict[int, CapturedScores] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            q = module._reshape_q(x)
            k, _ = module._reshape_kv(x)
            cos, sin = build_rope_cache(
                seq_len=x.size(1),
                head_dim=module.head_dim,
                base=module.cfg.attention.rope_base,
                device=x.device,
                dtype=x.dtype,
            )
            q, k = apply_rope(q, k, cos, sin)
            scale = 1.0 / math.sqrt(module.head_dim)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            captures[layer_idx] = CapturedScores(
                scores=scores.detach(),
                q=q.detach(),
                k=k.detach(),
            )

        return hook

    for layer_idx, block in enumerate(model.blocks):
        if layer_idx in wanted:
            handles.append(block.attn.register_forward_pre_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    missing = wanted - set(captures)
    if missing:
        raise RuntimeError(f"Did not capture requested layers: {sorted(missing)}")
    return captures


def _metrics_for_scores(scores: torch.Tensor, alpha: float) -> dict[str, float]:
    scaled = scores * alpha
    batch, heads, seq_len, _ = scaled.shape
    mask = torch.ones((seq_len, seq_len), device=scaled.device, dtype=torch.bool).tril()
    valid_logits = scaled[..., mask]

    masked = scaled.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked, dim=-1)
    probs_safe = probs.clamp_min(1e-12)
    entropy = -(probs * probs_safe.log()).sum(dim=-1)

    valid_counts = torch.arange(1, seq_len + 1, device=scaled.device).float()
    h_max = valid_counts.log().view(1, 1, seq_len)
    norm_entropy = entropy[..., 1:] / h_max[..., 1:]

    max_prob = probs.max(dim=-1).values
    top2 = masked[..., 1:, :].topk(k=2, dim=-1).values
    gap = top2[..., 0] - top2[..., 1]

    spectral = torch.linalg.matrix_norm(scaled.float(), ord=2, dim=(-2, -1))

    final_probs = probs[..., -1, :]
    final_entropy = -(final_probs * final_probs.clamp_min(1e-12).log()).sum(dim=-1)
    final_norm_entropy = final_entropy / math.log(seq_len)

    return {
        "score_l2_mean": float(spectral.mean().item()),
        "score_l2_max": float(spectral.max().item()),
        "logit_std": float(valid_logits.float().std(unbiased=False).item()),
        "absmax_p95": _quantile(valid_logits.abs(), 0.95),
        "gap_p95": _quantile(gap, 0.95),
        "entropy_mean": float(entropy.mean().item()),
        "norm_entropy_mean": float(norm_entropy.mean().item()),
        "final_norm_entropy_mean": float(final_norm_entropy.mean().item()),
        "max_prob_mean": float(max_prob.mean().item()),
        "frac_pmax_gt_0.9": float((max_prob > 0.9).float().mean().item()),
        "rows": float(batch * heads * seq_len),
    }


def _weight_stats(model: TransformerLM, layers: Iterable[int]) -> None:
    print("\nWeight/state summary")
    print(f"head_dim d_k: {model.cfg.head_dim}")
    for layer_idx in layers:
        attn = model.blocks[layer_idx].attn
        wq = attn.q_proj.weight.detach().float()
        wk = attn.k_proj.weight.detach().float()
        q_std = wq.std(unbiased=False).item()
        k_std = wk.std(unbiased=False).item()
        q_norm = torch.linalg.matrix_norm(wq, ord=2).item()
        k_norm = torch.linalg.matrix_norm(wk, ord=2).item()
        kwq_norm = torch.linalg.matrix_norm(wk @ wq.T, ord=2).item()
        print(
            f"layer={layer_idx} "
            f"std(WQ)={q_std:.6g} std(WK)={k_std:.6g} "
            f"||WQ||2={q_norm:.6g} ||WK||2={k_norm:.6g} "
            f"||WK WQ^T||2={kwq_norm:.6g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-zero calibration for attention score multipliers."
    )
    parser.add_argument("--config_path", default="configs/train/ALG.yaml")
    parser.add_argument("--overrides", default="{}")
    parser.add_argument("--layers", default="0,3,7")
    parser.add_argument("--alphas", default="1,1.25,1.5,2,3,4")
    parser.add_argument(
        "--batch-source",
        choices=("synthetic", "dataloader"),
        default="synthetic",
        help="Use synthetic tokens locally, or one real training batch on the cluster.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config_path, overrides=args.overrides)
    seed = int(getattr(cfg.trainer, "seed", 42))
    seed_everything(seed)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = TransformerLM(cfg.model).to(device)
    model.eval()

    batch_size = args.batch_size or int(getattr(cfg.data, "batch_size", 32))
    seq_len = args.seq_len or int(getattr(cfg.data, "seq_len", cfg.trainer.max_seq_len))

    if args.batch_source == "dataloader":
        batch = _prepare_dataloader_batch(cfg=cfg, device=device)
    else:
        batch = _make_synthetic_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=int(cfg.model.vocab_size),
            seed=seed,
            device=device,
        )

    input_ids = batch["input_ids"]
    layers = _parse_csv_ints(args.layers)
    alphas = _parse_csv_floats(args.alphas)

    if input_ids.size(1) != seq_len:
        seq_len = int(input_ids.size(1))

    print("Step-zero attention calibration")
    print(f"config: {args.config_path}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"batch_source: {args.batch_source}")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"layers: {layers}")
    print(f"alphas: {alphas}")

    _weight_stats(model, layers)
    captures = _capture_base_scores(model=model, input_ids=input_ids, layers=layers)

    columns = [
        "layer",
        "alpha",
        "score_l2_mean",
        "score_l2_max",
        "logit_std",
        "absmax_p95",
        "gap_p95",
        "entropy_mean",
        "norm_entropy_mean",
        "final_norm_entropy_mean",
        "max_prob_mean",
        "frac_pmax_gt_0.9",
        "std_ratio",
        "gap_ratio",
        "spectral_ratio",
    ]
    print("\n" + "\t".join(columns))
    for layer_idx in layers:
        base_metrics = _metrics_for_scores(captures[layer_idx].scores, 1.0)
        for alpha in alphas:
            metrics = _metrics_for_scores(captures[layer_idx].scores, alpha)
            row = {
                **metrics,
                "layer": layer_idx,
                "alpha": alpha,
                "std_ratio": metrics["logit_std"] / base_metrics["logit_std"],
                "gap_ratio": metrics["gap_p95"] / base_metrics["gap_p95"],
                "spectral_ratio": metrics["score_l2_mean"]
                / base_metrics["score_l2_mean"],
            }
            print(
                "\t".join(
                    f"{row[name]:.6g}" if isinstance(row[name], float) else str(row[name])
                    for name in columns
                )
            )


if __name__ == "__main__":
    main()
