from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from itertools import combinations
from functools import partial
from typing import Any


@dataclass
class LSHConfig:
    batch_size: int = 512
    k: int = 128
    b: int = 32
    max_bucket_size: int = 2000
    max_pairs_per_bucket: int = 2000


def band_key(sig: list[int], band_idx: int, r: int) -> str:
    start = band_idx * r
    bb = b"".join(
        int(v).to_bytes(8, "little", signed=True)
        for v in sig[start : start + r]
    )
    h = hashlib.blake2b(bb, digest_size=8).hexdigest()
    return f"{band_idx}:{h}"


def lsh_pairs_map_batches(
    batch: dict[str, Any],
    *,
    k: int,
    b: int,
    max_bucket_size: int,
    max_pairs_per_bucket: int,
) -> dict[str, list[Any]]:
    assert k % b == 0

    ids = batch["id"]
    sigs = batch["sig"]
    ok = batch.get("sig_ok", [True] * len(ids))

    r = k // b
    buckets: dict[str, list[Any]] = {}

    for _id, sig, good in zip(ids, sigs, ok):
        if not good or sig is None:
            continue

        try:
            sig_len = len(sig)
        except Exception:
            continue

        if sig_len != k:
            continue

        for band_idx in range(b):
            key = band_key(sig, band_idx, r)
            buckets.setdefault(key, []).append(_id)

    out_id1: list[Any] = []
    out_id2: list[Any] = []

    for key, bucket_ids in buckets.items():
        bucket_ids = sorted(set(bucket_ids))

        if len(bucket_ids) < 2 or len(bucket_ids) > max_bucket_size:
            continue

        seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(bucket_ids)

        pair_count = 0
        for a, c in combinations(bucket_ids, 2):
            out_id1.append(a)
            out_id2.append(c)
            pair_count += 1
            if pair_count >= max_pairs_per_bucket:
                break

    return {"id1": out_id1, "id2": out_id2}


def apply_pairs(ds, lsh_cfg: LSHConfig):
    batch_fn = partial(
        lsh_pairs_map_batches,
        k=lsh_cfg.k,
        b=lsh_cfg.b,
        max_bucket_size=lsh_cfg.max_bucket_size,
        max_pairs_per_bucket=lsh_cfg.max_pairs_per_bucket,
    )

    return (
        ds.select_columns(["id", "sig", "sig_ok"])
        .map_batches(
            batch_fn,
            batch_format="default",
            batch_size=lsh_cfg.batch_size,
        )
        .groupby(["id1", "id2"])
        .count()
        .drop_columns(["count()"])
    )