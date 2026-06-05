from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, rank: int = 0, deterministic: bool = True) -> None:
    """Seed all RNGs for reproducibility.

    Each DDP rank gets seed + rank so ranks produce different data/dropout
    patterns while still being deterministic given the same base seed.

    Args:
        seed: Base seed value.
        rank: DDP rank offset (0 for single-GPU).
        deterministic: If True, sets cuDNN to deterministic mode (slower but
            fully reproducible). Set False for benchmarking.
    """
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    """Return a DataLoader worker_init_fn that gives each worker a unique seed.

    Ensures numpy / random are re-seeded inside each worker process, which
    otherwise inherits the parent's RNG state without re-seeding.

    Usage:
        DataLoader(..., worker_init_fn=make_worker_init_fn(cfg.seed))
    """
    def _init(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init
