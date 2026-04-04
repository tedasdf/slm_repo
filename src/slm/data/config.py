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
class PreprocessStageConfig:
    version: str = 'v_1'
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

