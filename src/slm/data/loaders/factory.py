from __future__ import annotations

from ..config import DataLoaderConfig
from .text_loader import build_text_dataloaders
from .token_loader import build_token_dataloaders


def build_dataloaders(
    *,
    loader_cfg: DataLoaderConfig,
):
    mode = str(getattr(loader_cfg, "mode", "tokens")).strip().lower()

    if mode in {"text", "raw_text"}:
        return build_text_dataloaders(loader_cfg)

    if mode in {"tokens", "token", "token_blocks"}:
        return build_token_dataloaders(loader_cfg)

    raise ValueError(
        f"Unsupported loader mode={mode!r}. "
        "Use one of: 'text', 'raw_text', 'tokens', 'token_blocks'."
    )