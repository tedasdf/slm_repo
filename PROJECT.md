# SLM — Small Language Model Training Framework

A research-grade PyTorch pipeline for training, scaling-law analysis, and resource estimation of transformer language models. Designed for fixed-compute experiments on GPU/Slurm clusters with W&B integration.

---

## Repository Layout

```
slm_repo/
├── configs/                    # OmegaConf YAML configs
│   ├── train/
│   │   ├── baseline.yaml       # standard training run
│   │   └── smoke.yaml          # quick sanity-check run
│   ├── experiment/
│   │   └── scaling_law.yaml    # scaling-law sweep definition
│   └── data/
│       ├── preprocess.yaml     # preprocessing pipeline config
│       └── tokenizer/base.yaml # tokenizer config
├── src/slm/
│   ├── main.py                 # entry point (train or sweep)
│   ├── model/                  # transformer architecture
│   ├── training/               # trainer, optimizer, scheduler, callbacks
│   ├── data/                   # tokenizer, loaders, parquet conversion
│   ├── preprocess/             # data preprocessing pipeline
│   ├── experiments/            # scaling-law experiment runner
│   ├── resource_accounting/    # preflight + runtime resource estimation
│   └── utils/                  # shared config/path helpers
├── artifacts/                  # checkpoints, summaries (gitignored)
├── logs/
├── scripts/
├── pyproject.toml
└── Taskfile.yaml
```

---

## Modules

### `model/`

Transformer decoder (`TransformerLM`) built from composable, config-driven pieces.

| File | Contents |
|---|---|
| `config.py` | `ModelConfig`, `AttentionConfig`, `MLPConfig`, `InitConfig` |
| `model.py` | `TransformerLM` — full model with `count_params()` / `flops_per_token()` |
| `block.py` | `TransformerBlock` — pre-norm attn + MLP, aggregates resource accounting |
| `attention.py` | `CausalSelfAttention`, `SlidingWindowAttention` (SWA), `ExclusionSelfAttention` (XSA) |
| `mlp.py` | `GELUMLP`, `ReLU2MLP`, `SwiGLUMLP` |
| `norm.py` | `RMSNorm`, `LayerNorm` (wrapper around `nn.LayerNorm`) |
| `embeddings.py` | `TokenEmbedding` |
| `rope.py` | Rotary position embeddings |
| `registry.py` | Model registry / builder |

**Attention variants**

| Type | Key `attention_type` | Notes |
|---|---|---|
| Baseline / GQA | `"baseline"`, `"gqa"` | `CausalSelfAttention`; GQA when `num_kv_heads < num_heads` |
| Sliding Window | `"SlidingWindow"` | Attention bounded by `window_size` |
| Exclusion (XSA) | `"XSA"` | Projects out V-direction from output |

**Resource accounting** — every module exposes:
- `count_params() -> int` — exact param count via `.numel()`
- `flops_per_token(seq_len?) -> float` — analytic FLOPs (forward pass)

---

### `training/`

| File | Contents |
|---|---|
| `run_config.py` | `RunConfig` — top-level config composing all sub-configs |
| `trainer.py` | `Trainer` — train loop, eval, gradient accumulation, mixed precision |
| `builders.py` | Factory functions for model, optimizer, scheduler, data loaders |
| `callbacks.py` | `CallbackList`, callback protocol |
| `state.py` | `TrainState` — step, loss, elapsed time, best checkpoint tracking |
| `distributed.py` | DDP setup/teardown, `all_reduce_sum`, `barrier` |
| `logging.py` | W&B and print logging helpers |

**`RunConfig` structure**

```python
RunConfig
├── model:     ModelConfig       # architecture
├── trainer:   TrainerConfig     # steps, precision, grad accum, checkpointing
├── optimizer: OptimizerConfig   # AdamW lr, wd, betas, eps
├── scheduler: SchedulerConfig   # constant | cosine
├── logging:   LoggingConfig     # W&B project, run name, tags
├── tokenizer: TokenizerConfig
└── data:      DataLoaderConfig
```

**Entry point**

```bash
# single training run
python -m slm.main --config_path configs/train/baseline.yaml

# scaling-law sweep
python -m slm.main \
  --config_path configs/train/baseline.yaml \
  --experiment \
  --experiment_path configs/experiment/scaling_law.yaml
```

---

### `experiments/`

| File | Contents |
|---|---|
| `scaling_law.py` | `ScalingLawExperiment` — fixed-compute W&B sweep |
| `base.py` | Base experiment class |
| `callback.py` | Experiment-level callbacks |

**`ScalingLawExperiment`** sweeps over a grid of `(compute_budget, D:N ratio)` pairs.  
For each point it:
1. Derives target `N` and `D` from the Chinchilla formula (`C ≈ k·N·D`)
2. Selects the closest legal architecture from `layers_list × dim_num_heads`
3. Patches `RunConfig` (depth, width, `max_steps`) and launches a W&B agent

**`ScalingLawExperimentConfig`** fields

| Field | Description |
|---|---|
| `compute_list` | FLOPs budgets to sweep |
| `ratio_start/end/step` | D:N ratio grid |
| `layers_list` | Allowed layer counts |
| `dim_num_heads` | `{model_dim: num_heads}` map |
| `selection_param_mode` | `"kaplan"` / `"hoffman"` / `"exact"` |
| `train_flops_coeff` | `k` in `C = k·N·D` (default 6.0) |

---

### `resource_accounting/`

Preflight and runtime resource estimation. Each model module owns its own `count_params()` and `flops_per_token()`; this module orchestrates them.

| File | Contents |
|---|---|
| `estimator.py` | `ResourceEstimator` — delegates to model methods, adds memory/time estimates |
| `config.py` | `ResourceConfig` — GPU spec (TFLOPS, mem), MFU, budget fraction |
| `budget.py` | Memory budget checker |
| `reporter.py` | W&B + JSON summary writer |
| `callback.py` | Training callback for live metric sampling |
| `preprocess_hook.py` | `PreprocessResourceHook` — times preprocessing stages |

**Usage**

```python
from slm.model.model import TransformerLM
from slm.resource_accounting.estimator import ResourceEstimator

model = TransformerLM(model_cfg)          # CPU instance is fine
est = ResourceEstimator(model, trainer_cfg, resource_cfg)
result = est.estimate()                   # ResourceEstimate dataclass

# inspect per-component:
model.blocks[0].attn.count_params()
model.blocks[0].mlp.flops_per_token()
```

---

### `preprocess/`

Multi-stage data preprocessing pipeline with checkpoint/resume support.

**Stages** (in order)

| Stage | File | Description |
|---|---|---|
| `snapshot` | `stages/snapshot.py` | Pull raw dataset snapshot |
| `canonical` | `stages/canonical.py` | Normalise text fields |
| `quality_report` | `stages/quality_report.py` | Filter low-quality documents |
| `minihash` | `stages/minihash.py` | MinHash near-dedup |
| `pairs` | `stages/pairs.py` | Build dedup candidate pairs |
| `cluster_map` | `stages/cluster_map.py` | Connected-components clustering |
| `split` | `stages/split.py` | Train/val split |

**Pipeline internals**

| File | Contents |
|---|---|
| `pipeline/graph.py` | Stage dependency graph |
| `pipeline/planner.py` | Determines which stages need to run |
| `pipeline/runner.py` | Executes stages in order |
| `pipeline/checkpoints.py` | Checkpoint read/write |
| `pipeline/make_samples.py` | Tokenised sample generation |
| `io/` | Manifest, path, reader, writer helpers |

---

### `data/`

| File | Contents |
|---|---|
| `tokenizer.py` | `BPETokenizer` — train and encode |
| `tokenization.py` | Batch tokenization helpers |
| `loaders/text_loader.py` | Raw text dataloader |
| `loaders/token_loader.py` | Pre-tokenised token dataloader |
| `to_parquet.py` | Convert raw data to Parquet |
| `config.py` | `TokenizerConfig`, `DataLoaderConfig` |

---

## Configuration System

Configs are [OmegaConf](https://omegaconf.readthedocs.io/) structured dataclasses merged from YAML.  
`main.py` loads a `RunConfig` (training) and optionally a `ScalingLawExperimentConfig` (sweep).

```
configs/
├── train/baseline.yaml        ← override any RunConfig field
├── train/smoke.yaml           ← fast smoke-test config
├── experiment/scaling_law.yaml← ScalingLawExperimentConfig
└── data/preprocess.yaml       ← preprocessing pipeline settings
```

---

## Infrastructure

| Tool | Role |
|---|---|
| **W&B** | Experiment tracking, sweep orchestration |
| **DVC + S3** | Dataset and artifact versioning |
| **Slurm** | HPC job scheduling (M3 cluster) |
| **DDP** | Multi-GPU distributed training |
| **OmegaConf** | Typed, composable config merging |
