from dataclasses import dataclass
from pathlib import Path
import json
import itertools
from typing import Optional

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from config import PreprocessStageConfig
from tokenizers import Tokenizer
from tokenizers import models, trainers, pre_tokenizers, decoders


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        vocab = tokenizer.get_vocab()
        self.stoi = vocab
        self.itos = {i: tok for tok, i in vocab.items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    def token_to_id(self, token: str) -> int:
        token_id = self.tk.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Token not found in tokenizer vocab: {token}")
        return token_id

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()

    @classmethod
    def load(cls, path: Path):
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: Path):
        self.tk.save(str(path))


def train_tokenizer(
    text_iterator,
    vocab_size: int,
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    eos_token: str = "<eos>",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token],
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    return tokenizer

from omegaconf import OmegaConf

def load_config(config_path: str) -> PreprocessStageConfig:
    schema = OmegaConf.structured(PreprocessStageConfig)
    loaded_cfg = OmegaConf.load(config_path)
    merged = OmegaConf.merge(schema, loaded_cfg)
    missing = OmegaConf.missing_keys(merged)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    return OmegaConf.to_object(merged)


def resolve_config_paths(config_path: str) -> list[Path]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Expected a .yaml or .yml file, got: {path}")
        return [path]

    if path.is_dir():
        yaml_files = sorted(
            [
                p
                for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
            ]
        )
        if not yaml_files:
            raise ValueError(f"No YAML files found in directory: {path}")
        return yaml_files

    raise ValueError(f"Unsupported config path: {path}")


def iter_huggingface_texts(
    cfg: DatasetConfig,
    seed: int,
    smoke_test: bool = False,
):
    if cfg.source_type != "huggingface":
        raise ValueError(f"Expected source_type='huggingface', got {cfg.source_type}")

    ds = load_dataset(
        cfg.dataset_name,
        split=cfg.dataset_split,
        streaming=True,
        cache_dir=cfg.cache_dir,
    )

    ds = ds.shuffle(seed=seed, buffer_size=cfg.shuffle_buffer_size)

    max_docs = 1000 if smoke_test else None

    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break

        text = row.get(cfg.text_field)
        if not text:
            continue

        text = text.strip()
        if not text:
            continue

        yield text


def choose_bin_dtype(vocab_size: int):
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.uint32
    raise ValueError("Vocab too large for uint32 token storage.")


def train_or_load_tokenizer(
    dataset_cfg: DatasetConfig,
    tokenizer_cfg: TokenizerConfig,
    output_dir: Path,
    smoke_test: bool,
) -> BPETokenizer:
    vocab_path = output_dir / tokenizer_cfg.vocab_filename

    if tokenizer_cfg.reuse_existing and vocab_path.exists():
        tok = BPETokenizer.load(vocab_path)
        print(f"Loaded existing tokenizer from: {vocab_path}")
        return tok

    train_samples = 5_000 if smoke_test else dataset_cfg.tokenizer_train_samples

    text_iter = itertools.islice(
        iter_huggingface_texts(
            dataset_cfg,
            seed=dataset_cfg.seed,
            smoke_test=smoke_test,
        ),
        train_samples,
    )

    vocab = train_tokenizer(
        text_iter,
        tokenizer_cfg.vocab_size,
        unk_token=tokenizer_cfg.unk_token,
        eos_token=tokenizer_cfg.eos_token,
    )
    tok = BPETokenizer(vocab)
    tok.save(vocab_path)
    print(f"Saved tokenizer to: {vocab_path}")
    print(f"Tokenizer trained on up to {train_samples:,} streamed documents.")
    return tok


# def write_dataset_metadata(
#     cfg: DatasetConfig,
#     tokenizer_cfg: TokenizerConfig,
#     output_dir: Path,
#     stats: dict,
#     bin_dtype: str,
# ) -> None:
#     metadata = {
#         "dataset_name": cfg.dataset_name,
#         "source_type": cfg.source_type,
#         "dataset_split": cfg.dataset_split,
#         "text_field": cfg.text_field,
#         "seed": cfg.seed,
#         "shuffle_buffer_size": cfg.shuffle_buffer_size,
#         "target_train_tokens": cfg.target_train_tokens,
#         "target_val_tokens": cfg.target_val_tokens,
#         "tokenizer_train_samples": cfg.tokenizer_train_samples,
#         "actual_train_samples": stats["train_samples"],
#         "actual_val_samples": stats["val_samples"],
#         "actual_train_chars": stats["train_chars"],
#         "actual_val_chars": stats["val_chars"],
#         "actual_train_tokens": stats["train_tokens"],
#         "actual_val_tokens": stats["val_tokens"],
#         "bin_dtype": bin_dtype,
#         "tokenizer": {
#             "vocab_size": tokenizer_cfg.vocab_size,
#             "eos_token": tokenizer_cfg.eos_token,
#             "unk_token": tokenizer_cfg.unk_token,
#             "reuse_existing": tokenizer_cfg.reuse_existing,
#             "vocab_filename": tokenizer_cfg.vocab_filename,
#         },
#     }

#     metadata_path = output_dir / "metadata.json"
#     with metadata_path.open("w", encoding="utf-8") as f:
#         json.dump(metadata, f, indent=2)


def encode_and_write_bins(
    dataset_cfg: DatasetConfig,
    tokenizer_cfg: TokenizerConfig,
    tok: BPETokenizer,
    output_dir: Path,
    smoke_test: bool,
) -> dict:
    train_target = (
        min(dataset_cfg.target_train_tokens, 100_000)
        if smoke_test
        else dataset_cfg.target_train_tokens
    )
    val_target = (
        min(dataset_cfg.target_val_tokens, 20_000)
        if smoke_test
        else dataset_cfg.target_val_tokens
    )

    train_bin_path = output_dir / "train.bin"
    val_bin_path = output_dir / "val.bin"

    eos_id = tok.token_to_id(tokenizer_cfg.eos_token)
    bin_dtype = choose_bin_dtype(tok.vocab_size)

    stats = {
        "train_samples": 0,
        "val_samples": 0,
        "train_chars": 0,
        "val_chars": 0,
        "train_tokens": 0,
        "val_tokens": 0,
    }

    text_iter = iter_huggingface_texts(
        dataset_cfg,
        seed=dataset_cfg.seed + 1,
        smoke_test=smoke_test,
    )

    with train_bin_path.open("wb") as f_train, val_bin_path.open("wb") as f_val:
        for text in text_iter:
            ids = tok.encode(text)
            ids.append(eos_id)

            arr = np.asarray(ids, dtype=bin_dtype)
            n_tokens = int(arr.size)
            n_chars = len(text) + len(tokenizer_cfg.eos_token)

            if stats["val_tokens"] < val_target:
                arr.tofile(f_val)
                stats["val_tokens"] += n_tokens
                stats["val_chars"] += n_chars
                stats["val_samples"] += 1
                continue

            if stats["train_tokens"] < train_target:
                arr.tofile(f_train)
                stats["train_tokens"] += n_tokens
                stats["train_chars"] += n_chars
                stats["train_samples"] += 1

            if (
                stats["train_tokens"] >= train_target
                and stats["val_tokens"] >= val_target
            ):
                break

    stats["bin_dtype"] = np.dtype(bin_dtype).name
    return stats


def run_single_config(config_path: Path, smoke_test: bool) -> None:
    print("\n" + "=" * 80)
    print(f"Running config: {config_path}")
    print("=" * 80)

    app_cfg = load_config(str(config_path))
    dataset_cfg = app_cfg.dataset
    tokenizer_cfg = app_cfg.tokenizer

    output_dir = Path(dataset_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir:            {output_dir}")
    print(f"Target train tokens:   {dataset_cfg.target_train_tokens:,}")
    print(f"Target val tokens:     {dataset_cfg.target_val_tokens:,}")
    print(f"Tokenizer vocab size:  {tokenizer_cfg.vocab_size:,}")

    tok = train_or_load_tokenizer(
        dataset_cfg,
        tokenizer_cfg,
        output_dir,
        smoke_test=smoke_test,
    )

    stats = encode_and_write_bins(
        dataset_cfg,
        tokenizer_cfg,
        tok,
        output_dir,
        smoke_test=smoke_test,
    )

    # write_dataset_metadata(
    #     dataset_cfg,
    #     tokenizer_cfg,
    #     output_dir,
    #     stats,
    #     stats["bin_dtype"],
    # )

    print(f"Saved metadata to:     {output_dir / 'metadata.json'}")
    print(f"Actual train samples:  {stats['train_samples']:,}")
    print(f"Actual val samples:    {stats['val_samples']:,}")
    print(f"Actual train tokens:   {stats['train_tokens']:,}")
    print(f"Actual val tokens:     {stats['val_tokens']:,}")
    print(f"Token bin dtype:       {stats['bin_dtype']}")

    avg_train_tokens = (
        stats["train_tokens"] / stats["train_samples"]
        if stats["train_samples"] > 0
        else 0.0
    )
    print(f"Average train tokens/sample: {avg_train_tokens:.4f}")


def main(parser) -> None:
    config_paths = resolve_config_paths(parser.config_path)

    print(f"Found {len(config_paths)} config file(s).")

    for config_path in config_paths:
        run_single_config(config_path, smoke_test=parser.smoke_test)


if __name__ == "__main__":
    import argparse

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
    main(args)
