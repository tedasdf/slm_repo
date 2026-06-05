from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ..config import DataLoaderConfig
from ...utils.seed import make_worker_init_fn


def infer_token_dtype(bin_path: str | Path) -> np.dtype:
    """
    Simple heuristic:
    - if file was written as uint16, keep uint16
    - otherwise fall back to uint32

    If you already store dtype in a manifest, use that instead.
    """
    path = Path(bin_path)
    size = path.stat().st_size

    if size % np.dtype(np.uint16).itemsize == 0:
        return np.uint16
    return np.uint32


class TokenBlockDataset(Dataset):
    """Token-block dataset backed by a memory-mapped binary file.

    Memory model: np.memmap maps the file into virtual address space —
    pages are loaded on-demand by the OS, NOT all at once into RAM.
    A 20 GB bin file (e.g. 10B uint16 tokens) will only occupy as much
    physical RAM as the OS keeps hot in the page cache. There is no
    per-process memory limit from this class itself; the practical limit
    is the size of the bin file vs available page-cache headroom.
    Deferred: sharded multi-file loading (not blocking for current scale).
    """

    def __init__(
        self,
        bin_path: str | Path,
        seq_len: int,
        *,
        stride: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.dtype = dtype or infer_token_dtype(self.bin_path)

        self.tokens = np.memmap(self.bin_path, dtype=self.dtype, mode="r")

        if len(self.tokens) < self.seq_len + 1:
            raise ValueError(
                f"{self.bin_path} is too short for seq_len={self.seq_len}. "
                f"Need at least {self.seq_len + 1} tokens, got {len(self.tokens)}."
            )

        self.num_windows = 1 + (len(self.tokens) - (self.seq_len + 1)) // self.stride

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]
        if len(chunk) != self.seq_len + 1:
            raise IndexError(f"Invalid chunk at idx={idx}")

        chunk = np.asarray(chunk, dtype=np.int64)
        input_ids = torch.from_numpy(chunk[:-1].copy())
        targets = torch.from_numpy(chunk[1:].copy())

        return {
            "input_ids": input_ids,
            "targets": targets,
        }


def _build_token_torch_dataloaders(
    data_cfg: DataLoaderConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
) -> tuple[DataLoader, DataLoader | None]:
    train_dataset = TokenBlockDataset(
        bin_path=data_cfg.train_bin_path,
        seq_len=data_cfg.seq_len,
        stride=data_cfg.stride,
    )

    seed = getattr(data_cfg, "seed", 42)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=data_cfg.shuffle_train,
            drop_last=data_cfg.drop_last,
            seed=seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=(data_cfg.shuffle_train and train_sampler is None),
        sampler=train_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=data_cfg.drop_last,
        generator=generator,
        worker_init_fn=make_worker_init_fn(seed) if data_cfg.num_workers > 0 else None,
    )

    val_loader = None
    if data_cfg.val_bin_path is not None and Path(data_cfg.val_bin_path).exists():
        val_dataset = TokenBlockDataset(
            bin_path=data_cfg.val_bin_path,
            seq_len=data_cfg.seq_len,
            stride=data_cfg.seq_len,
        )

        val_sampler = None
        if is_distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader


def _build_token_ray_dataloaders(
    data_cfg: DataLoaderConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
):
    raise NotImplementedError(
        "Ray token loader is not implemented yet. "
        "Your current token-block memmap path is already efficient for local pretokenized data. "
        "Start with backend='torch' on the text loader first, then add a token Ray path later if you "
        "move token batches into a Ray-friendly store like Parquet on S3."
    )


def build_token_dataloaders(
    data_cfg: DataLoaderConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
):
    backend = str(getattr(data_cfg, "backend", "torch")).strip().lower()

    if backend == "torch":
        return _build_token_torch_dataloaders(
            data_cfg,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

    if backend == "ray":
        return _build_token_ray_dataloaders(
            data_cfg,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

    raise ValueError(
        f"Unsupported token loader backend={backend!r}. "
        "Use one of: 'torch', 'ray'."
    )