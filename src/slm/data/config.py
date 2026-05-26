from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    source_type: str = "huggingface"
    dataset_name: str = ""
    dataset_config_name: str = ""
    data_files_glob: str = ""

    # actual split names in the source dataset
    train_split_name: str = "train"
    val_split_name: Optional[str] = None
    test_split_name: Optional[str] = None

    # synthetic split settings for sources that do not have a real val split
    val_fraction: float = 0.0
    split_seed: int = 42

    # which text columns to read
    text_fields: list[str] = field(default_factory=lambda: ["text"])

    # dataset loading behavior
    cache_dir: Optional[str] = None
    streaming: bool = False

    # optional caps on raw samples loaded
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    # shuffle settings
    seed: int = 42
    shuffle: bool = True
    shuffle_buffer_size: int = 10_000


@dataclass
class TokenizerConfig:
    tokenizer_type: str = "bpe"
    vocab_size: int = 32_000
    min_frequency: int = 2

    pad_token: str = "<pad>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    bos_token: Optional[str] = None

    reuse_existing: bool = True
    tokenizer_train_samples: Optional[int] = 200_000

    # runtime / training
    tokenizer_path: Optional[str] = None
    allow_missing_tokenizer: bool = True

    @property
    def special_tokens(self) -> list[str]:
        tokens = [self.pad_token, self.eos_token, self.unk_token]
        if self.bos_token is not None:
            tokens.append(self.bos_token)
        return tokens

@dataclass
class PreprocessConfig:
    # sequence construction
    block_size: int = 1024
    packing: bool = True
    truncation: bool = True
    stride: Optional[int] = None

    # token handling
    append_eos: bool = True
    drop_short_sequences: bool = True

    # token budgets
    target_train_tokens: int = 100_000_000
    target_val_tokens: int = 5_000_000
    target_test_tokens: int = 0
    shard_target_bytes: int = 256 * 1024 * 1024

    # create splits from train if missing
    val_fraction: float = 0.0
    split_seed: int = 42

    # execution
    num_proc: int = 4
    debug: bool = False
    
@dataclass
class DataLoaderConfig:
    mode: str = "tokens"
    backend: str = "torch"

    seq_len: int = 1024
    batch_size: int = 8
    shuffle_train: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True
    stride: Optional[int] = None
    ray_prefetch_batches: int = 1

    # token mode
    train_bin_path: Optional[str] = None
    val_bin_path: Optional[str] = None

    # text mode — local files
    source_type: Optional[str] = None
    train_paths: Optional[str] = None
    val_paths: Optional[str] = None

    # text mode — huggingface
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_split_name: str = "train"
    val_split_name: Optional[str] = None
    streaming: bool = True
    cache_dir: Optional[str] = None

    # text mode — shared
    text_fields: list[str] = field(default_factory=lambda: ["text"])
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = 2000
    val_fraction: float = 0.0
    split_seed: int = 42
    seed: int = 42
    shuffle: bool = True
    shuffle_buffer_size: int = 10000

    ray_num_cpus: int = 4
    ray_read_concurrency: int = 4
    ray_override_num_blocks: Optional[int] = None
  

@dataclass
class PreprocessStageConfig:
    version: str = 'v_1'
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

