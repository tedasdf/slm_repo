from __future__ import annotations


from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .run_config import DataLoaderConfig
from ..data.config import DatasetConfig
from ..data.tokenization import extract_text, hash_to_unit_interval, iter_examples, stable_example_key

class RawTextDataset(IterableDataset):
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        split_name: str,
        *,
        seed_offset: int = 0,
        max_samples: int | None = None,
        val_fraction: float = 0.0,
        split_seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        is_distributed: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.split_name = split_name
        self.seed_offset = seed_offset
        self.max_samples = max_samples
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed

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

    def _iter_examples(self):
        yielded = 0

        source_split = (
            self.dataset_cfg.train_split_name
            if self._use_synthetic_val_split()
            else self.split_name
        )

        for row in iter_examples(
            self.dataset_cfg,
            split_name=source_split,
            seed_offset=self.seed_offset,
        ):
            if not self._row_belongs_to_requested_split(row):
                continue

            text = extract_text(row, self.dataset_cfg.text_fields)
            if text is None:
                continue

            if isinstance(text, str):
                text = text.strip()

            if not text:
                continue

            yield text
            yielded += 1

            if self.max_samples is not None and yielded >= self.max_samples:
                break

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if self.is_distributed:
            total_shards = self.world_size * num_workers
            shard_id = self.rank * num_workers + worker_id
        else:
            total_shards = num_workers
            shard_id = worker_id

        base_iterator = self._iter_examples()

        for idx, sample in enumerate(base_iterator):
            if idx % total_shards != shard_id:
                continue
            yield sample

def collate_text_batch(batch: list[str]) -> dict[str, list[str]]:
    return {"text": batch}

def build_text_dataloaders(
    dataset_cfg: DatasetConfig,
    loader_cfg: DataLoaderConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
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
        rank=rank,
        world_size=world_size,
        is_distributed=is_distributed,
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
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
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