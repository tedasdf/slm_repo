from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    source_type: str = "huggingface"
    dataset_name: str = ""
    split_names: list[str] = field(default_factory=lambda: ["train", "validation"])
    text_fields: list[str] = field(default_factory=lambda: ["text"])
    cache_dir: Optional[str] = None
    streaming: bool = False
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
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
    block_size: int = 1024
    packing: bool = True
    truncation: bool = True
    stride: Optional[int] = None
    append_eos: bool = True
    drop_short_sequences: bool = True
    target_train_tokens: int = 100_000_000
    target_val_tokens: int = 5_000_000
    num_proc: int = 4
    debug: bool = False


@dataclass
class PreprocessStageConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)