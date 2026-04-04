from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .run_config import DataLoaderConfig


def infer_token_dtype(bin_path: str | Path) -> np.dtype:
    """
    Simple heuristic:
    - if file was written as uint16, keep uint16
    - otherwise fall back to uint32

    If you already store dtype in a manifest, use that instead.
    """
    path = Path(bin_path)
    size = path.stat().st_size

    # crude fallback heuristic
    if size % np.dtype(np.uint16).itemsize == 0:
        return np.uint16
    return np.uint32


class TokenBlockDataset(Dataset):
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


def build_dataloaders(data_cfg: DataLoaderConfig) -> tuple[DataLoader, DataLoader | None]:
    train_dataset = TokenBlockDataset(
        bin_path=data_cfg.train_bin_path,
        seq_len=data_cfg.seq_len,
        stride=data_cfg.stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=data_cfg.shuffle_train,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=data_cfg.drop_last,
    )

    val_loader = None
    if data_cfg.val_bin_path is not None and Path(data_cfg.val_bin_path).exists():
        val_dataset = TokenBlockDataset(
            bin_path=data_cfg.val_bin_path,
            seq_len=data_cfg.seq_len,
            stride=data_cfg.seq_len,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg.batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader