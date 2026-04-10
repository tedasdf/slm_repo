from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re
import ray
from omegaconf import OmegaConf
from ray.data import DataContext

from ..stages.canonical import CanonicalizerConfig, apply_canonicalize
from ..stages.minihash import MinHashConfig, apply_minhash
try:
    import wandb
except ImportError:
    wandb = None


# =========================
# Config
# =========================

@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "slm-preprocess"
    entity: str | None = None
    name: str | None = None
    mode: str = "offline"   # online | offline | disabled
    tags: list[str] = field(default_factory=lambda: ["canonicalize"])
    log_code: bool = False


@dataclass
class RunConfig:
    input_dir: str = "dataset/processed_datasets/unified_python"
    canonicalize_output_dir: str | None = None
    minhash_output_dir: str | None = None

    debug: bool = False
    debug_max_rows: int = 2000

    canonicalize_parseable_only: bool = False
    minhash_sig_only: bool = False

    input_columns: list[str] | None = None
    sample_size: int | None = None
    object_store_memory_bytes: int | None = 2684354560
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)
    save_local_metrics_json: bool = True


@dataclass
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    canonicalize: CanonicalizerConfig = field(default_factory=CanonicalizerConfig)
    minhash: MinHashConfig = field(default_factory=MinHashConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_pipeline_config(cfg_path: str) -> PipelineConfig:
    schema = OmegaConf.structured(PipelineConfig)
    loaded = OmegaConf.load(cfg_path)
    merged = OmegaConf.merge(schema, loaded)
    return OmegaConf.to_object(merged)


# =========================
# Helpers
# =========================
def infer_sample_size(path: str) -> int | None:
    name = Path(path).name
    match = re.search(r"sample_(\d+)", name)
    if match:
        return int(match.group(1))
    return None

def read_input(run_cfg: RunConfig) -> ray.data.Dataset:
    read_kwargs: dict[str, Any] = {}
    if run_cfg.input_columns:
        read_kwargs["columns"] = run_cfg.input_columns
    return ray.data.read_parquet(run_cfg.input_dir, **read_kwargs)


def maybe_limit_debug(ds: ray.data.Dataset, run_cfg: RunConfig) -> ray.data.Dataset:
    if run_cfg.debug:
        return ds.limit(run_cfg.debug_max_rows)
    return ds


def maybe_filter_parseable(ds: ray.data.Dataset, run_cfg: RunConfig) -> ray.data.Dataset:
    if run_cfg.canonicalize_parseable_only:
        return ds.filter(lambda row: row.get("parse_ok", False))
    return ds

def get_canonicalize_output_dir(run_cfg) -> str:
    if run_cfg.canonicalize_output_dir:
        return run_cfg.canonicalize_output_dir
    return os.path.join(run_cfg.input_dir, "stages", "01_canonicalize")


def get_minhash_output_dir(run_cfg) -> str:
    if run_cfg.minhash_output_dir:
        return run_cfg.minhash_output_dir
    return os.path.join(run_cfg.input_dir, "stages", "02_minhash")


def maybe_filter_sig_ok(ds: ray.data.Dataset, run_cfg) -> ray.data.Dataset:
    if getattr(run_cfg, "minhash_sig_only", False):
        return ds.filter(lambda row: row.get("sig_ok", False))
    return ds

def configure_ray_data_context() -> None:
    # Must happen before creating the Dataset.
    ctx = DataContext.get_current()         # control execution/ UI settings 
    ctx.enable_rich_progress_bars = True    #  use the new rich progress bar
    ctx.use_ray_tqdm = False                # avoids tqdm based progress


def maybe_init_wandb(cfg: PipelineConfig):
    if not cfg.wandb.enabled:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. pip install wandb")

    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,   # online / offline / disabled
        config={
            "run": {
                "input_dir": cfg.run.input_dir,
                "output_dir": cfg.run.canonicalize_output_dir,
                "debug": cfg.run.debug,
                "debug_max_rows": cfg.run.debug_max_rows,
                "canonicalize_parseable_only": cfg.run.canonicalize_parseable_only,
                "input_columns": cfg.run.input_columns,
                "object_store_memory_bytes": cfg.run.object_store_memory_bytes,
            },
            "canonicalize": {
                "representation": str(cfg.canonicalize.representation),
                "traversal": str(cfg.canonicalize.traversal),
                "remove_docstrings": cfg.canonicalize.remove_docstrings,
                "rename_locals": cfg.canonicalize.rename_locals,
                "rename_args": cfg.canonicalize.rename_args,
                "rename_function_names": cfg.canonicalize.rename_function_names,
                "rename_class_names": cfg.canonicalize.rename_class_names,
                "normalize_literals": str(cfg.canonicalize.normalize_literals),
                "keep_builtins": cfg.canonicalize.keep_builtins,
                "max_code_chars": cfg.canonicalize.max_code_chars,
                "on_parse_error": str(cfg.canonicalize.on_parse_error),
            },
        },
        tags=cfg.wandb.tags,
    )


def write_local_metrics(metrics: dict[str, Any], out_dir: str) -> None:
    metrics_path = Path(out_dir) / "canonicalize_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# =========================
# Main
# =========================
def run_preprocess(cfg_path: str) -> tuple[str, str]:
    t0_total = time.perf_counter()

    cfg = load_pipeline_config(cfg_path)
    run_cfg = cfg.run

    canonicalize_out_dir = get_canonicalize_output_dir(run_cfg)
    minhash_out_dir = get_minhash_output_dir(run_cfg)

    Path(canonicalize_out_dir).mkdir(parents=True, exist_ok=True)
    Path(minhash_out_dir).mkdir(parents=True, exist_ok=True)

    wb_run = maybe_init_wandb(cfg)

    ray_init_kwargs = dict(run_cfg.ray_init_kwargs or {})
    if run_cfg.object_store_memory_bytes is not None:
        ray_init_kwargs.setdefault(
            "object_store_memory",
            run_cfg.object_store_memory_bytes,
        )

    ray.init(**ray_init_kwargs)

    try:
        configure_ray_data_context()

        ds = read_input(run_cfg)
        ds = maybe_limit_debug(ds, run_cfg)

        ds = apply_canonicalize(ds, cfg.canonicalize)
        ds = maybe_filter_parseable(ds, run_cfg)

        # write 01_canonicalize
        t0_can = time.perf_counter()
        ds.write_parquet(canonicalize_out_dir)
        t_can = time.perf_counter() - t0_can

        ds = apply_minhash(ds, cfg.minhash)
        ds = maybe_filter_sig_ok(ds, run_cfg)

        # write 02_minhash
        t0_mh = time.perf_counter()
        ds.write_parquet(minhash_out_dir)
        t_mh = time.perf_counter() - t0_mh

        total_time = time.perf_counter() - t0_total

        metrics = {
            "status": "success",
            "canonicalize_write_exec_sec": t_can,
            "minhash_write_exec_sec": t_mh,
            "total_wallclock_sec": total_time,
            "canonicalize_output_dir": canonicalize_out_dir,
            "minhash_output_dir": minhash_out_dir,
        }

        if wb_run is not None:
            sample_size = run_cfg.sample_size
            if sample_size is None:
                sample_size = infer_sample_size(run_cfg.input_dir)

            metrics["sample_size"] = sample_size
            wb_run.config.update({"sample_size": sample_size}, allow_val_change=True)
            wb_run.log(metrics)
            wb_run.summary.update(metrics)

        if run_cfg.save_local_metrics_json:
            write_local_metrics(metrics, minhash_out_dir)

        print(f"[canonicalize] wrote parquet to: {canonicalize_out_dir}")
        print(f"[minhash] wrote parquet to: {minhash_out_dir}")
        return canonicalize_out_dir, minhash_out_dir

    finally:
        if wb_run is not None:
            wb_run.finish()
        ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocess stages.")
    parser.add_argument("--config", type=str, default="configs/data/preprocess.yaml")
    args = parser.parse_args()

    run_preprocess(args.config)