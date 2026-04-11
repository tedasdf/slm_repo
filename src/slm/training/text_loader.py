from __future__ import annotations

import hashlib
from typing import Iterator, Optional

from torch.utils.data import DataLoader, IterableDataset

from .run_config import DataLoaderConfig
from ..data.config import DatasetConfig
from ..data.tokenization import extract_text, hash_to_unit_interval, iter_examples, stable_example_key





class RawTextDataset(IterableDataset):
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        split_name: str,
        *,
        seed_offset: int,
        max_samples: Optional[int] = None,
        val_fraction: float = 0.0,
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.split_name = split_name
        self.seed_offset = seed_offset
        self.max_samples = max_samples
        self.val_fraction = val_fraction
        self.split_seed = split_seed

    def _use_synthetic_val_split(self) -> bool:
        return (
            self.dataset_cfg.val_split_name is None
            and self.val_fraction > 0
            and self.split_name in {"train", "val"}
        )

    def _row_belongs_to_requested_split(self, row: dict) -> bool:
        if not self._use_synthetic_val_split():
            return True

        key = stable_example_key(row, self.dataset_cfg.text_fields)
        u = hash_to_unit_interval(key, self.split_seed)
        assigned = "val" if u < self.val_fraction else "train"
        return assigned == self.split_name

    def __iter__(self) -> Iterator[str]:
        use_synth_split = self._use_synthetic_val_split()

        # If we are synthesizing val from train, both train and val must read
        # from the train source and then filter deterministically.
        source_split_name = (
            self.dataset_cfg.train_split_name if use_synth_split else self.split_name
        )

        # Important: when synthesizing val, do not pass max_samples into iter_examples,
        # otherwise you cap the raw stream before filtering and may get too few examples.
        source_max_samples = None if use_synth_split else self.max_samples

        yielded = 0
        for row in iter_examples(
            self.dataset_cfg,
            split_name=source_split_name,
            seed=self.dataset_cfg.seed + self.seed_offset,
            smoke_test=False,
            max_samples=source_max_samples,
        ):
            if not self._row_belongs_to_requested_split(row):
                continue

            text = extract_text(row, self.dataset_cfg.text_fields)
            if not text:
                continue

            yield text
            yielded += 1

            if self.max_samples is not None and yielded >= self.max_samples:
                break


def collate_text_batch(batch: list[str]) -> dict[str, list[str]]:
    return {"text": batch}

def build_text_dataloaders(
    dataset_cfg: DatasetConfig,
    loader_cfg: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader | None]:
    val_fraction = getattr(dataset_cfg, "val_fraction", 0.0)
    split_seed = getattr(dataset_cfg, "split_seed", 42)

    train_dataset = RawTextDataset(
        dataset_cfg=dataset_cfg,
        split_name="train" if dataset_cfg.val_split_name is None else dataset_cfg.train_split_name,
        seed_offset=1,
        max_samples=dataset_cfg.max_train_samples,
        val_fraction=val_fraction,
        split_seed=split_seed,
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
    has_validation = dataset_cfg.val_split_name is not None or val_fraction > 0

    if has_validation:
        val_dataset = RawTextDataset(
            dataset_cfg=dataset_cfg,
            split_name="val" if dataset_cfg.val_split_name is None else dataset_cfg.val_split_name,
            seed_offset=2,
            max_samples=dataset_cfg.max_val_samples,
            val_fraction=val_fraction,
            split_seed=split_seed,
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