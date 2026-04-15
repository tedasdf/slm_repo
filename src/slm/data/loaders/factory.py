from __future__ import annotations

from ..config import DataLoaderConfig
from .text_loader import build_text_dataloaders
from .token_loader import build_token_dataloaders

from __future__ import annotations

from ..config import DataLoaderConfig
from .text_loader import build_text_dataloaders
from .token_loader import build_token_dataloaders


def build_dataloaders(
    *,
    loader_cfg: DataLoaderConfig,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
):
    mode = str(getattr(loader_cfg, "mode", "tokens")).strip().lower()

    if mode in {"text", "raw_text"}:
        return build_text_dataloaders(
            loader_cfg,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

    if mode in {"tokens", "token", "token_blocks"}:
        return build_token_dataloaders(
            loader_cfg,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

    raise ValueError(
        f"Unsupported loader mode={mode!r}. "
        "Use one of: 'text', 'raw_text', 'tokens', 'token_blocks'."
    )