from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import hashlib
import gzip
import os
from pathlib import Path
from typing import Iterator, Optional
import glob
import json
import random
import zstandard as zstd
import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.slm.utils.config import resolve_config_paths
from src.slm.utils.paths import finish_run, start_run

from .tokenizer import BPETokenizer
from .config import PreprocessStageConfig

@dataclass
class PreprocessArtifactPaths:
    tokenizer_path: str
    splits_dir: str


# helper functions
def load_config(config_path: str) -> PreprocessStageConfig:
    schema = OmegaConf.structured(PreprocessStageConfig)
    loaded_cfg = OmegaConf.load(config_path)
    merged = OmegaConf.merge(schema, loaded_cfg)

    missing = OmegaConf.missing_keys(merged)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")

    return OmegaConf.to_object(merged)

def prepare_preprocess_artifacts(run_dir: str) -> PreprocessArtifactPaths:
    splits_dir = os.path.join(run_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    return PreprocessArtifactPaths(
        tokenizer_path=os.path.join(run_dir, "tokenizer.json"),
        splits_dir=splits_dir,
    )

def choose_bin_dtype(vocab_size: int):
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.uint32
    raise ValueError("Vocab too large for uint32 token storage.")


# iterate through text
def extract_text(row: dict, text_fields: list[str]) -> Optional[str]:
    parts: list[str] = []
    for field_name in text_fields:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                parts.append(cleaned)
        else:
            cleaned = str(value).strip()
            if cleaned:
                parts.append(cleaned)

    if not parts:
        return None
    return "\n".join(parts)

def iter_huggingface_examples(
    dataset_cfg,
    split_name: str,
    seed: int,
    smoke_test: bool = False,
    max_samples: Optional[int] = None,
):
    if dataset_cfg.source_type != "huggingface":
        raise ValueError(f"Expected source_type='huggingface', got {dataset_cfg.source_type}")

    ds = load_dataset(
        dataset_cfg.dataset_name,
        name=dataset_cfg.dataset_config_name,
        split=split_name,
        streaming=dataset_cfg.streaming,
        cache_dir=dataset_cfg.cache_dir,
    )
    if dataset_cfg.shuffle:
        if dataset_cfg.streaming:
            ds = ds.shuffle(seed=seed, buffer_size=dataset_cfg.shuffle_buffer_size)
        else:
            ds = ds.shuffle(seed=seed)

    hard_cap = 1000 if smoke_test else max_samples

    count = 0
    for row in ds:
        if hard_cap is not None and count >= hard_cap:
            break
        count += 1
        yield row

def iter_examples(
    dataset_cfg,
    split_name: str,
    seed: int,
    smoke_test: bool = False,
    max_samples: Optional[int] = None,
):
    if dataset_cfg.source_type == "dolma_local":
        yield from iter_dolma_json_gz_examples(
            dataset_cfg=dataset_cfg,
            seed=seed,
            smoke_test=smoke_test,
            max_samples=max_samples,
        )
        return

    if dataset_cfg.source_type != "huggingface":
        raise ValueError(f"Unsupported source_type: {dataset_cfg.source_type}")
    
    os.environ["DATA_DIR"] = 'dolma'
    
    ds = load_dataset(
        dataset_cfg.dataset_name,
        split=split_name,
        streaming=dataset_cfg.streaming,
        cache_dir=dataset_cfg.cache_dir,
    )

    if dataset_cfg.shuffle:
        if dataset_cfg.streaming:
            ds = ds.shuffle(seed=seed, buffer_size=dataset_cfg.shuffle_buffer_size)
        else:
            ds = ds.shuffle(seed=seed)

    hard_cap = 1000 if smoke_test else max_samples

    count = 0
    for row in ds:
        if hard_cap is not None and count >= hard_cap:
            break
        count += 1
        yield row


def iter_dolma_json_gz_examples(
    dataset_cfg,
    seed: int,
    smoke_test: bool = False,
    max_samples: Optional[int] = None,
):
    pattern = dataset_cfg.data_files_glob
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise ValueError(f"No Dolma files matched: {pattern}")

    if dataset_cfg.shuffle:
        rng = random.Random(seed)
        rng.shuffle(paths)

    hard_cap = 1000 if smoke_test else max_samples
    count = 0

    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if hard_cap is not None and count >= hard_cap:
                    return
                if not line.strip():
                    continue

                row = json.loads(line)
                count += 1
                yield row

# use text -> hash to determin the key -> validation/train
def stable_example_key(row: dict, text_fields: list[str]) -> str:
    for key in ("id", "uuid", "guid", "doc_id"):
        value = row.get(key)
        if value is not None:
            return str(value)

    text = extract_text(row, text_fields)
    if text is None:
        return ""
    return text

def hash_to_unit_interval(key: str, seed: int) -> float:
    raw = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    value = int(digest[:8], 16)
    return value / 0xFFFFFFFF

def assign_train_row_to_split(row: dict, dataset_cfg, preprocess_cfg) -> str:
    key = stable_example_key(row, dataset_cfg.text_fields)
    u = hash_to_unit_interval(key, preprocess_cfg.split_seed)

    if preprocess_cfg.val_fraction > 0 and u < preprocess_cfg.val_fraction:
        return "val"
    return "train"



# split text into valdiation train test
def iter_final_split_texts(
    cfg: PreprocessStageConfig,
    role: str,
    smoke_test: bool = False,
):
    if role not in {'train','val','test'}:
        raise ValueError(f"Unknown role: {role}")
    
    dataset_cfg = cfg.dataset
    preprocess_cfg = cfg.preprocess

    if role == "val" and dataset_cfg.val_split_name is not None:
        for row in iter_examples(
            dataset_cfg,
            split_name=dataset_cfg.val_split_name,
            seed=dataset_cfg.seed + 2,
            smoke_test=smoke_test,
            max_samples=dataset_cfg.max_val_samples,
        ):
            text = extract_text(row, dataset_cfg.text_fields)
            if text:
                yield text
        return

    if role == "test":
        if dataset_cfg.test_split_name is None:
            return
        for row in iter_examples(
            dataset_cfg,
            split_name=dataset_cfg.test_split_name,
            seed=dataset_cfg.seed + 3,
            smoke_test=smoke_test,
            max_samples=dataset_cfg.max_test_samples,
        ):
            text = extract_text(row, dataset_cfg.text_fields)
            if text:
                yield text
        return

    # source train split
    for row in iter_examples(
        dataset_cfg,
        split_name=dataset_cfg.train_split_name,
        seed=dataset_cfg.seed + 1,
        smoke_test=smoke_test,
        max_samples=dataset_cfg.max_train_samples,
    ):
        text = extract_text(row, dataset_cfg.text_fields)
        if not text:
            continue

        if dataset_cfg.val_split_name is not None:
            if role == "train":
                yield text
            continue

        assigned = assign_train_row_to_split(row, dataset_cfg, preprocess_cfg)

        if role == assigned:
            yield text



# tokenizer functions
def train_tokenizer(
    text_iterator: Iterator[str],
    vocab_size: int,
    special_tokens: list[str],
    unk_token: str,
    min_frequency: int,
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    return tokenizer

def train_or_load_tokenizer(
    cfg: PreprocessStageConfig,
    tokenizer_path: Path,
    smoke_test: bool,
) -> BPETokenizer:
    tokenizer_cfg = cfg.tokenizer

    if tokenizer_cfg.reuse_existing and tokenizer_path.exists():
        tok = BPETokenizer.load(tokenizer_path)
        print(f"Loaded existing tokenizer from: {tokenizer_path}")
        return tok

    tokenizer_train_samples = tokenizer_cfg.tokenizer_train_samples
    if smoke_test:
        tokenizer_train_samples = min(tokenizer_train_samples or 5_000, 5_000)

    text_iter = itertools.islice(
        iter_final_split_texts(cfg, role="train", smoke_test=smoke_test),
        tokenizer_train_samples,
    )

    vocab = train_tokenizer(
        text_iter,
        vocab_size=tokenizer_cfg.vocab_size,
        special_tokens=tokenizer_cfg.special_tokens,
        unk_token=tokenizer_cfg.unk_token,
        min_frequency=tokenizer_cfg.min_frequency,
    )
    tok = BPETokenizer(vocab)
    tok.save(tokenizer_path)
    print(f"Saved tokenizer to: {tokenizer_path}")
    print(f"Tokenizer trained on up to {tokenizer_train_samples:,} documents.")
    return tok



#### Writing bin files #####
def _adjust_targets(preprocess_cfg, smoke_test: bool) -> tuple[int, int, int]:
    train_target = preprocess_cfg.target_train_tokens
    val_target = preprocess_cfg.target_val_tokens
    test_target = getattr(preprocess_cfg, "target_test_tokens", None) or 0

    if smoke_test:
        train_target = min(train_target, 100_000)
        val_target = min(val_target, 20_000)
        test_target = min(test_target, 20_000)

    return train_target, val_target, test_target

def _write_encoded(arr: np.ndarray, handle, token_key: str, char_key: str, sample_key: str, stats: dict, n_chars: int) -> None:
    arr.tofile(handle)
    stats[token_key] += int(arr.size)
    stats[char_key] += n_chars
    stats[sample_key] += 1

def encode_and_write_bins(
    cfg: PreprocessStageConfig,
    tok: BPETokenizer,
    artifacts: PreprocessArtifactPaths,
    smoke_test: bool,
) -> dict:
    tokenizer_cfg = cfg.tokenizer
    preprocess_cfg = cfg.preprocess

    train_target, val_target, test_target = _adjust_targets(preprocess_cfg, smoke_test)

    train_path = Path(artifacts.splits_dir) / "train"
    val_path = Path(artifacts.splits_dir) / "val"
    test_path = Path(artifacts.splits_dir) / "test"

    eos_id = tok.token_to_id(tokenizer_cfg.eos_token)
    bin_dtype = choose_bin_dtype(tok.vocab_size)

    stats = {
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
        "train_chars": 0,
        "val_chars": 0,
        "test_chars": 0,
        "train_tokens": 0,
        "val_tokens": 0,
        "test_tokens": 0,
        "bin_dtype": np.dtype(bin_dtype).name,
    }

    def write_split(text_iter, split_dir: Path, token_key, char_key, sample_key, target_tokens):
        if target_tokens <= 0:
            return

        split_dir.mkdir(parents=True, exist_ok=True)

        shard_target_bytes = getattr(preprocess_cfg, "shard_target_bytes", 256 * 1024 * 1024)
        split_name = split_dir.name

        shard_idx = 0
        shard_bytes = 0
        f = None

        def open_new_shard(idx: int):
            path = split_dir / f"{split_name}_{idx:06d}.bin"
            return path.open("wb")

        try:
            f = open_new_shard(shard_idx)

            for text in text_iter:
                ids = tok.encode(text)
                if preprocess_cfg.append_eos:
                    ids.append(eos_id)
                if not ids:
                    continue

                arr = np.asarray(ids, dtype=bin_dtype)
                sample_bytes = arr.nbytes

                if shard_bytes > 0 and shard_bytes + sample_bytes > shard_target_bytes:
                    f.close()
                    shard_idx += 1
                    f = open_new_shard(shard_idx)
                    shard_bytes = 0

                n_chars = len(text) + (
                    len(tokenizer_cfg.eos_token) if preprocess_cfg.append_eos else 0
                )
                _write_encoded(arr, f, token_key, char_key, sample_key, stats, n_chars)
                shard_bytes += sample_bytes

                if stats[token_key] >= target_tokens:
                    break
        finally:
            if f is not None:
                f.close()

    write_split(
        iter_final_split_texts(cfg, role="train", smoke_test=smoke_test),
        train_path,
        "train_tokens", "train_chars", "train_samples",
        train_target,
    )

    write_split(
        iter_final_split_texts(cfg, role="val", smoke_test=smoke_test),
        val_path,
        "val_tokens", "val_chars", "val_samples",
        val_target,
    )

    if getattr(cfg.dataset, "test_split_name", None) is not None and test_target > 0:
        write_split(
            iter_final_split_texts(cfg, role="test", smoke_test=smoke_test),
            test_path,
            "test_tokens", "test_chars", "test_samples",
            test_target,
        )

    return stats



def run_single_config(config_path: Path, smoke_test: bool) -> None:
    print("\n" + "=" * 80)
    print(f"Running config: {config_path}")
    print("=" * 80)
    
    cfg = load_config(str(config_path))

    if (
        cfg.dataset.val_split_name is None
        and cfg.preprocess.target_val_tokens > 0
        and cfg.preprocess.val_fraction <= 0
    ):
        raise ValueError(
            "val_split_name is None, but target_val_tokens > 0 and val_fraction <= 0. "
            "Either provide an explicit validation split or set preprocess.val_fraction > 0."
        )


    ctx = start_run(
        config_path=str(config_path),
        process_type="tokenizer",
        dataset_version_id=cfg.version,
    )

    try:
        
        artifact_paths = prepare_preprocess_artifacts(ctx.run_dir)

        
        print(f"Train split:           {cfg.dataset.train_split_name}")
        print(f"Val split:             {cfg.dataset.val_split_name}")
        print(f"Test split:            {cfg.dataset.test_split_name}")
        print(f"Target train tokens:   {cfg.preprocess.target_train_tokens:,}")
        print(f"Target val tokens:     {cfg.preprocess.target_val_tokens:,}")
        print(f"Tokenizer vocab size:  {cfg.tokenizer.vocab_size:,}")

        tok = train_or_load_tokenizer(cfg, tokenizer_path=Path(artifact_paths.tokenizer_path), smoke_test=smoke_test)
        
        stats = encode_and_write_bins(
            cfg=cfg,
            tok=tok,
            artifacts=artifact_paths,
            smoke_test=smoke_test,
        )
        
        finish_run(
            ctx.manifest_path,
            status="completed",
            artifacts=asdict(artifact_paths),
            stats=stats,
        )
        return ctx.run_dir
    except Exception as e:
        finish_run(ctx.manifest_path, status="failed", extras={"error": str(e)})
        raise

            

def main(parser) -> None:
    # find paths 
    config_paths = resolve_config_paths(parser.config_path)
    print(f"Found {len(config_paths)} config file(s).")

    run_dirs = []
    for config_path in config_paths:
        # individual config 
        run_dir = run_single_config(config_path, smoke_test=parser.smoke_test)
        run_dirs.append(run_dir)

    for run_dir in run_dirs:
        print(f"RUN_DIR={run_dir}")

if __name__ == "__main__":

    ####
    # parser argument parser
    ####
    parser = argparse.ArgumentParser(description="Dataset preprocessing pipeline")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/tokenizer.yaml",
        help="Path to a YAML config file or a directory containing YAML configs",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a small quick preprocessing sample",
    )
    args = parser.parse_args()
    
    # 
    # main
    #
    main(args)
