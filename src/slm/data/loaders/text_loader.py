from __future__ import annotations

from typing import Iterator, Optional, Any

from torch.utils.data import DataLoader, IterableDataset

from ..config import DatasetConfig, DataLoaderConfig
from ..tokenization import (
    extract_text,
    hash_to_unit_interval,
    iter_examples,
    stable_example_key,
)


def _use_synthetic_val_split(
    dataset_cfg: DatasetConfig,
    split_name: str,
    val_fraction: float,
) -> bool:
    return (
        dataset_cfg.val_split_name is None
        and val_fraction > 0
        and split_name in {"train", "val"}
    )


def _row_belongs_to_requested_split(
    *,
    row: dict,
    dataset_cfg: DatasetConfig,
    split_name: str,
    val_fraction: float,
    split_seed: int,
) -> bool:
    if not _use_synthetic_val_split(dataset_cfg, split_name, val_fraction):
        return True

    key = stable_example_key(row, dataset_cfg.text_fields)
    u = hash_to_unit_interval(key, split_seed)
    assigned = "val" if u < val_fraction else "train"
    return assigned == split_name


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
        return _use_synthetic_val_split(
            self.dataset_cfg,
            self.split_name,
            self.val_fraction,
        )

    def _row_belongs_to_requested_split(self, row: dict) -> bool:
        return _row_belongs_to_requested_split(
            row=row,
            dataset_cfg=self.dataset_cfg,
            split_name=self.split_name,
            val_fraction=self.val_fraction,
            split_seed=self.split_seed,
        )

    def __iter__(self) -> Iterator[str]:
        use_synth_split = self._use_synthetic_val_split()

        source_split_name = (
            self.dataset_cfg.train_split_name
            if use_synth_split
            else self.split_name
        )

        # When synthesizing val, do not cap the raw stream before filtering.
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


def _build_text_torch_dataloaders(
    dataset_cfg: DatasetConfig,
    loader_cfg: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader | None]:
    val_fraction = getattr(dataset_cfg, "val_fraction", 0.0)
    split_seed = getattr(dataset_cfg, "split_seed", 42)

    train_dataset = RawTextDataset(
        dataset_cfg=dataset_cfg,
        split_name=(
            "train"
            if dataset_cfg.val_split_name is None
            else dataset_cfg.train_split_name
        ),
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
            split_name=(
                "val"
                if dataset_cfg.val_split_name is None
                else dataset_cfg.val_split_name
            ),
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


def _require_ray():
    try:
        import ray  # noqa: F401
        return ray
    except ImportError as exc:
        raise ImportError(
            "Ray backend requested, but 'ray' is not installed. "
            "Install it first, or switch loader_cfg.backend='torch'."
        ) from exc


def _resolve_ray_paths(dataset_cfg: DatasetConfig, split_name: str) -> Any:
    """
    Best-effort path resolution for file-backed Ray reads.

    Supported patterns:
    - dataset_cfg.data_files = {"train": ..., "val": ...}
    - dataset_cfg.train_data_files / val_data_files
    - dataset_cfg.train_data_files_glob / val_data_files_glob
    - dataset_cfg.data_files_glob
    """
    data_files = getattr(dataset_cfg, "data_files", None)
    if isinstance(data_files, dict) and split_name in data_files:
        return data_files[split_name]

    candidates = [
        f"{split_name}_data_files",
        f"{split_name}_data_files_glob",
        "data_files_glob",
        "data_path",
        "data_paths",
    ]

    for name in candidates:
        value = getattr(dataset_cfg, name, None)
        if value:
            return value

    raise ValueError(
        f"Could not resolve file paths for Ray text loading for split={split_name!r}. "
        "Add one of: data_files, <split>_data_files, <split>_data_files_glob, or data_files_glob."
    )


def _build_ray_text_dataset(
    dataset_cfg: DatasetConfig,
    *,
    split_name: str,
    max_samples: Optional[int],
    val_fraction: float,
    split_seed: int,
):
    ray = _require_ray()

    use_synth_split = _use_synthetic_val_split(dataset_cfg, split_name, val_fraction)
    source_split_name = (
        dataset_cfg.train_split_name if use_synth_split else split_name
    )

    source_type = str(getattr(dataset_cfg, "source_type", "jsonl")).strip().lower()
    paths = _resolve_ray_paths(dataset_cfg, source_split_name)

    text_fields = list(getattr(dataset_cfg, "text_fields", []) or [])
    read_columns = text_fields if source_type == "parquet" and text_fields else None

    if source_type == "parquet":
        ds = ray.data.read_parquet(paths, columns=read_columns)
    elif source_type in {"json", "jsonl"}:
        ds = ray.data.read_json(paths)
    elif source_type in {"text", "txt"}:
        ds = ray.data.read_text(paths)
    else:
        raise NotImplementedError(
            f"Ray text loader does not support source_type={source_type!r} yet. "
            "Use loader_cfg.backend='torch' for this dataset source."
        )

    def keep_row(row: dict) -> bool:
        if not _row_belongs_to_requested_split(
            row=row,
            dataset_cfg=dataset_cfg,
            split_name=split_name,
            val_fraction=val_fraction,
            split_seed=split_seed,
        ):
            return False

        text = extract_text(row, dataset_cfg.text_fields)
        return bool(text)

    def to_text_row(row: dict) -> dict[str, str]:
        return {"text": extract_text(row, dataset_cfg.text_fields)}

    ds = ds.filter(keep_row).map(to_text_row)

    if max_samples is not None:
        ds = ds.limit(max_samples)

    return ds


class _RayTextBatchLoader:
    def __init__(
        self,
        ds,
        *,
        batch_size: int,
        drop_last: bool,
        prefetch_batches: int = 1,
    ) -> None:
        self.ds = ds
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.prefetch_batches = prefetch_batches

    def __iter__(self):
        for batch in self.ds.iter_batches(
            batch_size=self.batch_size,
            batch_format="numpy",
            drop_last=self.drop_last,
            prefetch_batches=self.prefetch_batches,
        ):
            texts = batch["text"]
            if hasattr(texts, "tolist"):
                texts = texts.tolist()
            else:
                texts = list(texts)

            yield {"text": texts}


def _build_text_ray_dataloaders(
    dataset_cfg: DatasetConfig,
    loader_cfg: DataLoaderConfig,
):
    val_fraction = getattr(dataset_cfg, "val_fraction", 0.0)
    split_seed = getattr(dataset_cfg, "split_seed", 42)
    prefetch_batches = int(getattr(loader_cfg, "ray_prefetch_batches", 1))

    train_split_name = (
        "train"
        if dataset_cfg.val_split_name is None
        else dataset_cfg.train_split_name
    )

    train_ds = _build_ray_text_dataset(
        dataset_cfg,
        split_name=train_split_name,
        max_samples=dataset_cfg.max_train_samples,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    train_loader = _RayTextBatchLoader(
        train_ds,
        batch_size=loader_cfg.batch_size,
        drop_last=loader_cfg.drop_last,
        prefetch_batches=prefetch_batches,
    )

    val_loader = None
    has_validation = dataset_cfg.val_split_name is not None or val_fraction > 0

    if has_validation:
        val_split_name = (
            "val"
            if dataset_cfg.val_split_name is None
            else dataset_cfg.val_split_name
        )

        val_ds = _build_ray_text_dataset(
            dataset_cfg,
            split_name=val_split_name,
            max_samples=dataset_cfg.max_val_samples,
            val_fraction=val_fraction,
            split_seed=split_seed,
        )

        val_loader = _RayTextBatchLoader(
            val_ds,
            batch_size=loader_cfg.batch_size,
            drop_last=False,
            prefetch_batches=prefetch_batches,
        )

    return train_loader, val_loader


def build_text_dataloaders(
    dataset_cfg: DatasetConfig,
    loader_cfg: DataLoaderConfig,
):
    backend = str(getattr(loader_cfg, "backend", "torch")).strip().lower()

    if backend == "torch":
        return _build_text_torch_dataloaders(dataset_cfg, loader_cfg)

    if backend == "ray":
        return _build_text_ray_dataloaders(dataset_cfg, loader_cfg)

    raise ValueError(
        f"Unsupported text loader backend={backend!r}. "
        "Use one of: 'torch', 'ray'."
    )