import hashlib
from dataclasses import dataclass
from itertools import combinations
import ray
import random


@dataclass
class LSHConfig:
    batch_size: int = 512
    k: int = 128
    b: int = 32
    max_bucket_size: int = 2000
    max_pairs_per_bucket: int = 2000


def band_key(sig, band_idx, r):
    start = band_idx * r
    bb = b"".join(
        int(v).to_bytes(8, "little", signed=True) for v in sig[start : start + r]
    )
    h = hashlib.blake2b(bb, digest_size=8).hexdigest()
    return f"{band_idx}:{h}"


def lsh_pairs_map_batches(
    batch, k=128, b=32, max_bucket_size=2000, max_pairs_per_bucket=2000
):
    assert k % b == 0
    ids = batch["id"]
    sigs = batch["sig"]
    ok = batch.get("sig_ok", [True] * len(ids))

    r = k // b
    buckets = {}

    for _id, sig, good in zip(ids, sigs, ok):
        if not good:
            continue
        if sig is None:
            continue
        # handle list/tuple/np.ndarray uniformly
        try:
            sig_len = len(sig)
        except Exception:
            continue
        if sig_len != k:
            continue

        for band_idx in range(b):
            key = band_key(sig, band_idx, r)
            buckets.setdefault(key, []).append(_id)

    out_id1, out_id2 = [], []
    for bucket_ids in buckets.values():
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


def run_stage_lsh(cfg, stage_paths: dict[str, str], ds_minihash=None):
    """
    Produces candidate pairs (id1, id2) from minhash signatures.
    Writes to stage_paths["lsh"] which maps to 03_pairs/.
    """
    if ds_minihash is None:
        ds_minihash = ray.data.read_parquet(stage_paths["minhash"])
        if getattr(cfg.run, "debug", False):
            ds_minihash = ds_minihash.limit(getattr(cfg.run, "debug_max_rows", 2000))
    ds_minihash = ds_minihash.filter(lambda r: r.get("sig_ok", False))

    def lsh_pairs_map_batches_safe(batch, k, b, max_bucket_size, max_pairs_per_bucket):
        try:
            return lsh_pairs_map_batches(
                batch,
                k=k,
                b=b,
                max_bucket_size=max_bucket_size,
                max_pairs_per_bucket=max_pairs_per_bucket,
            )
        except Exception as e:
            print("LSH UDF error:", type(e).__name__, e)
            return {"id1": [], "id2": []}

    pairs_ds = (
        ds_minihash.select_columns(["id", "sig", "sig_ok"])
        .map_batches(
            lsh_pairs_map_batches_safe,
            batch_format="default",
            batch_size=cfg.lsh.batch_size,
            fn_kwargs={
                "k": cfg.lsh.k,
                "b": cfg.lsh.b,
                "max_bucket_size": cfg.lsh.max_bucket_size,
                "max_pairs_per_bucket": cfg.lsh.max_pairs_per_bucket,
            },
        )
        # remove duplicates across batches
        .groupby(["id1", "id2"])
        .count()
        .drop_columns(["count()"])
    )

    out_dir = stage_paths["pairs"]
    pairs_ds.write_parquet(out_dir)
    print("wrote pairs:", out_dir, "| rows:", pairs_ds.count())

    return pairs_ds