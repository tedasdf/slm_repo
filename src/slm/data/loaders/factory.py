from __future__ import annotations

from typing import Optional

from ..config import DatasetConfig, DataLoaderConfig
from .text_loader import build_text_dataloaders
from .token_loader import build_token_dataloaders


def build_dataloaders(
    *,
    loader_cfg: DataLoaderConfig,
    dataset_cfg: Optional[DatasetConfig] = None,
):
    """
    Single entry point for constructing train/val loaders.

    Routing rules:
    - mode=text   -> text_loader.py
    - mode=tokens -> token_loader.py

    If mode is not provided, we infer it from whether train_bin_path exists.
    """
    mode = getattr(loader_cfg, "mode", None)
    if mode is None:
        mode = "tokens" if getattr(loader_cfg, "train_bin_path", None) else "text"

    mode = str(mode).strip().lower()

    if mode in {"text", "raw_text"}:
        if dataset_cfg is None:
            raise ValueError(
                "dataset_cfg is required when loader mode is 'text'."
            )
        return build_text_dataloaders(dataset_cfg, loader_cfg)

    if mode in {"tokens", "token", "token_blocks"}:
        return build_token_dataloaders(loader_cfg)

    raise ValueError(
        f"Unsupported loader mode={mode!r}. "
        f"Use one of: 'text', 'raw_text', 'tokens', 'token_blocks'."
    )