from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataLoaderConfig:
    train_bin_path: str = "artifacts/tokenizer/latest/splits/train.bin"
    val_bin_path: Optional[str] = "artifacts/tokenizer/latest/splits/val.bin"

    seq_len: int = 1024
    batch_size: int = 8

    stride: Optional[int] = None   # None -> use seq_len
    shuffle_train: bool = True

    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True