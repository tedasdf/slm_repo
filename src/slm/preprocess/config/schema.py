from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from omegaconf import OmegaConf


from dataclasses import dataclass, field
from typing import Any

from ..stages.canonical import CanonicalizerConfig


@dataclass
class RunConfig:
    input_dir: str = "dataset/processed_datasets/unified_python"
    canonicalize_output_dir: str | None = None

    debug: bool = False
    debug_max_rows: int = 2000

    canonicalize_parseable_only: bool = False

    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    canonicalize: CanonicalizerConfig = field(default_factory=CanonicalizerConfig)


def load_pipeline_config(cfg_path: str) -> PipelineConfig:
    schema = OmegaConf.structured(PipelineConfig)
    loaded = OmegaConf.load(cfg_path)
    merged = OmegaConf.merge(schema, loaded)
    return OmegaConf.to_object(merged)
