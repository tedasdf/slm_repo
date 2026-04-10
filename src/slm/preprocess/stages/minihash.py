import hashlib
import struct
from typing import List
from dataclasses import dataclass

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
    return hv & I64_MAX  # force into signed int64 range


def shingles_from_node_types(node_types: List[str], n: int) -> set[str]:
    if not node_types or len(node_types) < n:
        return set()
    return {"|".join(node_types[i : i + n]) for i in range(len(node_types) - n + 1)}


def minhash_signature_from_shingles(
    shingle_set: set[str], cfg: MinHashConfig
) -> List[int]:
    if not shingle_set:
        return [I64_MAX] * cfg.num_perm

    sig = [I64_MAX] * cfg.num_perm
    for s in shingle_set:
        for i in range(cfg.num_perm):
            hv = _h63(s, cfg.seed0 + i)
            if hv < sig[i]:
                sig[i] = hv
    return sig


def minhash_from_node_types(node_types: List[str], cfg: MinHashConfig) -> List[int]:
    S = shingles_from_node_types(node_types, n=cfg.shingle_n)
    return minhash_signature_from_shingles(S, cfg)


def make_add_minhash(cfg: MinHashConfig):
    # closure captures cfg cleanly for Ray
    def add_minhash(row):
        node_types = row.get("node_types") or []
        if (not row.get("parse_ok")) or (len(node_types) < cfg.shingle_n):
            row["sig_ok"] = False
            row["sig"] = [I64_MAX] * cfg.num_perm
            return row

        row["sig_ok"] = True
        row["sig"] = minhash_from_node_types(node_types, cfg)
        return row

    return add_minhash



def run_stage_minhash(cfg, stage_paths: dict[str, str], ds_canon=None):
    """
    If ds_canon is provided (from same run), use it.
    Otherwise read canonicalize output from disk.
    """
    if ds_canon is None:
        ds_canon = ray.data.read_parquet(stage_paths["canonicalize"])
        if getattr(cfg.run, "debug", False):
            ds_canon = ds_canon.limit(getattr(cfg.run, "debug_max_rows", 2000))

    # (canonicalize stage already filtered parse_ok, but safe to keep this)
    ds_canon = ds_canon.filter(lambda r: r.get("parse_ok", False))

    ds_sig = ds_canon.map(make_add_minhash(cfg.minhash))
    ds_minihash = ds_sig.filter(lambda r: r.get("sig_ok", False))

    out_dir = stage_paths["minhash"]
    ds_minihash.write_parquet(out_dir)
    print("wrote minhash:", out_dir, "| rows:", ds_minihash.count())

    return ds_minihash


if __name__ == "__main__":
    import ray

    ray.init()

    in_dir = "intermediate/canon_node_types_v0001_single"
    ds_canon = ray.data.read_parquet(in_dir)

    # Keep parseable only (optional; you can also keep and mark sig_ok False)
    ds_canon = ds_canon.filter(lambda r: r.get("parse_ok", False))

    cfg = MinHashConfig(shingle_n=5, num_perm=128, seed0=1337)

    ds_sig = ds_canon.map(make_add_minhash(cfg))

    rows = ds_sig.take(3)
    for r in rows:
        print(r["id"], "sig_ok=", r["sig_ok"], "sig_head=", r["sig"][:8])

    out_dir = "intermediate/minhash_node_v0001"
    ds2_single = ds_sig.repartition(1)  # one output shard
    ds2_single.write_parquet(out_dir)