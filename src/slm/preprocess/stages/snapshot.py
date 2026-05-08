import os
import ray
import pandas as pd
from .utils import _stable_md5

from dataclasses import dataclass
from typing import Literal


@dataclass
class DedupConfig:
    mode: Literal["exact", "cluster"] = "cluster"
    key_col: str | None = None  # override grouping key if you want
    snapshot_keep_cols: list[str] | None = None


def dedup_snapshot(ds, mode: str, key_col: str | None = None):
    if mode not in {"exact", "cluster"}:
        raise ValueError("mode must be 'exact' or 'cluster'")

    if key_col is None:
        key_col = "canon_hash" if mode == "exact" else "cluster_id"

    # Keep only rows with a non-empty key
    ds_k = ds.filter(lambda r: bool(r.get(key_col)))

    def pick_best_group(pdf: "pd.DataFrame") -> "pd.DataFrame":
        # Deterministic representative choice:
        # 1) longest code_ref
        # 2) then longest instruction
        # 3) then smallest md5(id)
        code_len = (
            pdf.get("code_ref", pd.Series([""] * len(pdf)))
            .fillna("")
            .astype(str)
            .str.len()
        )
        instr_len = (
            pdf.get("instruction", pd.Series([""] * len(pdf)))
            .fillna("")
            .astype(str)
            .str.len()
        )
        id_hash = (
            pdf.get("id", pd.Series([""] * len(pdf)))
            .fillna("")
            .astype(str)
            .map(_stable_md5)
        )

        pdf = pdf.assign(_code_len=code_len, _instr_len=instr_len, _id_hash=id_hash)
        pdf = pdf.sort_values(
            by=["_code_len", "_instr_len", "_id_hash"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        return pdf.head(1).drop(columns=["_code_len", "_instr_len", "_id_hash"])

    reps = ds_k.groupby(key_col).map_groups(pick_best_group, batch_format="pandas")
    return reps, key_col


def run_stage_snapshot(
    cfg, stage_paths: dict[str, str], ds_minihash=None, ds_cluster_map=None
):
    """
    Inputs:
      - stage_paths["minhash"]      (02_minhash/)
      - stage_paths["cluster_map"]  (04_cluster_map/)
    Output:
      - stage_paths["snapshot"]     (05_snapshot/)
    """

    if ds_minihash is None:
        ds_minihash = ray.data.read_parquet(stage_paths["minhash"])
        if getattr(cfg.run, "debug", False):
            ds_minihash = ds_minihash.limit(getattr(cfg.run, "debug_max_rows", 2000))

    if ds_cluster_map is None:
        ds_cluster_map = ray.data.read_parquet(
            stage_paths["cluster_map"]
        )  # id, cluster_id
        if getattr(cfg.run, "debug", False):
            ds_cluster_map = ds_cluster_map.limit(
                getattr(cfg.run, "debug_max_rows", 2000)
            )

    ds_with_cluster = ds_minihash.join(
        ds_cluster_map,
        on=("id",),
        join_type="left_outer",
        num_partitions=200,  # tune: 50–500 depending on dataset size
    )

    def fill_singletons(row):
        if not row.get("cluster_id"):
            row["cluster_id"] = _stable_md5(str(row["id"]))
        return row

    ds_with_cluster = ds_with_cluster.map(fill_singletons)

    reps, used_key = dedup_snapshot(
        ds_with_cluster,
        mode=cfg.snapshot.mode,
        key_col=cfg.snapshot.key_col,
    )

    default_keep = [
        "dataset",
        "id",
        "instruction",
        "code_ref",
        "language",
        "has_code",
        "cluster_id",
    ]
    keep_cols = default_keep
    cols = reps.columns()  # Ray will try to fetch schema
    if not cols:
        # fallback: sample one row if dataset non-empty
        sample = reps.take(1)
        cols = list(sample[0].keys()) if sample else []

    reps_out = reps.select_columns([c for c in keep_cols if c in cols])

    snapshot_dir = stage_paths["snapshot"]
    os.makedirs(snapshot_dir, exist_ok=True)
    reps_out.write_parquet(snapshot_dir)

    print("wrote pairs:", snapshot_dir, "| rows:", reps_out.count())

    return reps_out