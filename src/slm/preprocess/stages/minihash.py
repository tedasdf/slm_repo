from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import partial
from typing import Any


I64_MAX = (1 << 63) - 1


@dataclass
class MinHashConfig:
    shingle_n: int = 5
    num_perm: int = 128
    seed0: int = 1337


def _h63(s: str, seed: int) -> int:
    h = hashlib.blake2b(digest_size=8, person=seed.to_bytes(8, "little"))
    h.update(s.encode("utf-8", errors="ignore"))
    hv = struct.unpack("<Q", h.digest())[0]
    return hv & I64_MAX


def shingles_from_node_types(node_types: list[str], n: int) -> set[str]:
    if not node_types or len(node_types) < n:
        return set()
    return {"|".join(node_types[i : i + n]) for i in range(len(node_types) - n + 1)}


def minhash_signature_from_shingles(
    shingle_set: set[str],
    cfg: MinHashConfig,
) -> list[int]:
    if not shingle_set:
        return [I64_MAX] * cfg.num_perm

    sig = [I64_MAX] * cfg.num_perm
    for s in shingle_set:
        for i in range(cfg.num_perm):
            hv = _h63(s, cfg.seed0 + i)
            if hv < sig[i]:
                sig[i] = hv
    return sig


def minhash_from_node_types(node_types: list[str], cfg: MinHashConfig) -> list[int]:
    shingles = shingles_from_node_types(node_types, n=cfg.shingle_n)
    return minhash_signature_from_shingles(shingles, cfg)


def transform_minhash_row(
    row: dict[str, Any],
    minhash_cfg: MinHashConfig,
) -> dict[str, Any]:
    node_types = row.get("node_types") or []

    if (not row.get("parse_ok", False)) or (len(node_types) < minhash_cfg.shingle_n):
        row["sig_ok"] = False
        row["sig"] = [I64_MAX] * minhash_cfg.num_perm
        return row

    row["sig_ok"] = True
    row["sig"] = minhash_from_node_types(node_types, minhash_cfg)
    return row


def apply_minhash(ds, minhash_cfg: MinHashConfig):
    row_fn = partial(transform_minhash_row, minhash_cfg=minhash_cfg)
    return ds.map(row_fn)


def only_sig_ok(ds):
    return ds.filter(lambda row: row.get("sig_ok", False))