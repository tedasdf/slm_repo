import ray
import os

from dataclasses import dataclass
from typing import Dict
from .utils import _stable_md5


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class ClusterConfig:
    pairs_batch_rows: int = 200_000  # reduce if driver RAM tight
    map_write_rows: int = 1_000_000  # rows per parquet write chunk


def run_stage_cluster_map(
    cfg,
    stage_paths: dict[str, str],
    pairs_ds=None,
    stable_cluster_ids: bool = True,
):
    """
    Input:  stage_paths["pairs"] parquet with columns (id1, id2)
    Output: stage_paths["cluster_map"] parquet with columns (id, cluster_id)
    """

    if pairs_ds is None:
        pairs_ds = ray.data.read_parquet(stage_paths["pairs"]).select_columns(
            ["id1", "id2"]
        )
        if getattr(cfg.run, "debug", False):
            pairs_ds = pairs_ds.limit(getattr(cfg.run, "debug_max_rows", 2000))

    # 1) Build Union-Find by streaming batches (no take_all)
    uf = UnionFind()
    for batch in pairs_ds.iter_batches(
        batch_size=cfg.cluster.pairs_batch_rows,
        batch_format="pyarrow",
    ):
        id1 = batch["id1"].to_pylist()
        id2 = batch["id2"].to_pylist()
        for a, b in zip(id1, id2):
            if a and b:
                uf.union(str(a), str(b))

    ids_in_pairs = list(uf.parent.keys())
    print("UF ids seen in pairs:", len(ids_in_pairs))

    # 2) Root -> cluster_id (stable if using min-id per component)
    if stable_cluster_ids:
        root_to_min = {}
        for _id in ids_in_pairs:
            r = uf.find(_id)
            cur = root_to_min.get(r)
            if cur is None or _id < cur:
                root_to_min[r] = _id
        root_to_cid = {r: _stable_md5(min_id) for r, min_id in root_to_min.items()}
    else:
        root_to_cid = {}
        for _id in ids_in_pairs:
            r = uf.find(_id)
            if r not in root_to_cid:
                root_to_cid[r] = _stable_md5(r)

    # 3) Write cluster_map parquet in chunks
    cluster_map_dir = stage_paths["cluster_map"]
    os.makedirs(cluster_map_dir, exist_ok=True)

    rows = []
    wrote_parts = 0
    for _id in ids_in_pairs:
        r = uf.find(_id)
        rows.append({"id": _id, "cluster_id": root_to_cid[r]})

        if len(rows) >= cfg.cluster.map_write_rows:
            ray.data.from_items(rows).write_parquet(
                os.path.join(cluster_map_dir, f"part_{wrote_parts:05d}")
            )
            wrote_parts += 1
            rows = []

    if rows:
        ray.data.from_items(rows).write_parquet(
            os.path.join(cluster_map_dir, f"part_{wrote_parts:05d}")
        )

    print("wrote cluster_map:", cluster_map_dir)
    return ray.data.read_parquet(cluster_map_dir)