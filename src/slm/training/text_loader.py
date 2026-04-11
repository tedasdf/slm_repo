from __future__ import annotations

from typing import Iterator, Optional

from torch.utils.data import DataLoader, IterableDataset

from .run_config import DataLoaderConfig
from ..data.config import DatasetConfig
from ..data.tokenization import extract_text, iter_examples


class RawTextDataset(IterableDataset):
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        split_name: str,
        *,
        seed_offset: int,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.split_name = split_name
        self.seed_offset = seed_offset
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[str]:
        for row in iter_examples(
            self.dataset_cfg,
            split_name=self.split_name,
            seed=self.dataset_cfg.seed + self.seed_offset,
            smoke_test=False,
            max_samples=self.max_samples,
        ):
            text = extract_text(row, self.dataset_cfg.text_fields)
            if text:
                yield text


def collate_text_batch(batch: list[str]) -> dict[str, list[str]]:
    return {"text": batch}


def build_text_dataloaders(
    dataset_cfg: DatasetConfig,
    loader_cfg: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader | None]:
    train_dataset = RawTextDataset(
        dataset_cfg=dataset_cfg,
        split_name=dataset_cfg.train_split_name,
        seed_offset=1,
        max_samples=dataset_cfg.max_train_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=loader_cfg.pin_memory,
        drop_last=loader_cfg.drop_last,
        collate_fn=collate_text_batch,
    )

    val_loader = None
    if dataset_cfg.val_split_name is not None:
        val_dataset = RawTextDataset(
            dataset_cfg=dataset_cfg,
            split_name=dataset_cfg.val_split_name,
            seed_offset=2,
            max_samples=dataset_cfg.max_val_samples,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=loader_cfg.pin_memory,
            drop_last=False,
            collate_fn=collate_text_batch,
        )

    return train_loader, val_loader