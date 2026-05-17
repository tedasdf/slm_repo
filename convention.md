# Repo and TAP Conventions for Multi-Algorithm Experiment Tracking

## Core Idea

TAP should not directly understand every algorithm deeply.

Each ML repo should expose a common experiment contract.

TAP supports many algorithm families by reading that contract.

The common shape is:

```text
experiment
→ config
→ launch command
→ status
→ metrics
→ artifacts
→ evaluation
```

---

## Main Principle

```text
Repo = does the ML work

TAP = tracks, launches, compares, and connects runs

Config = contract between repo and TAP

Artifacts = how one run feeds another

Registry = what combinations are valid
```

---

## Standard Repo Structure

Every ML repo should follow a similar external structure:

```text
repo_name/
  configs/
    train/
    eval/
    sweep/
    presets/

  src/
    package_name/
      main.py
      config/
      data/
      models/
      training/
      evaluation/
      tracking/
      utils/

  scripts/
    slurm/
    local/
    export/

  tests/

  artifacts/        # gitignored
  logs/             # gitignored
  checkpoints/      # gitignored
  wandb/            # gitignored

  pyproject.toml
  README.md
  .env.example
  .gitignore
```

Examples:

```text
slm_repo/
  src/slm/

diffusion_repo/
  src/diffusion/

rl_repo/
  src/rl_agent/
```

Each repo can be different internally, but the outside interface should look similar.

---

## One Entrypoint Convention

Every repo should have one main command style.

For SLM:

```bash
PYTHONPATH=src python -m slm.main --config_path configs/train/local_smoke.yaml
```

For diffusion:

```bash
PYTHONPATH=src python -m diffusion.main --config_path configs/train/smoke.yaml
```

For RL:

```bash
PYTHONPATH=src python -m rl_agent.main --config_path configs/train/cartpole_smoke.yaml
```

This means TAP only needs to know:

```text
repo path
python module
config path
launch command
```

TAP should not need to understand every internal implementation detail.

---

## Standard Config Shape

Every config should use consistent top-level sections:

```yaml
experiment:
  name: "local-smoke"
  family: "slm"
  task: "language_modeling"
  tags: ["smoke", "local"]

model:
  ...

algorithm:
  ...

data:
  ...

training:
  ...

evaluation:
  ...

runtime:
  ...

tracking:
  ...

artifacts:
  ...
```

Each family can still have specific fields, but the top-level structure should stay consistent.

---

## SLM Config Example

```yaml
experiment:
  family: "slm"
  task: "causal_language_modeling"

model:
  type: "decoder_transformer"
  attention_type: "baseline"

algorithm:
  objective: "next_token_prediction"

data:
  source_type: "dolma_local"

training:
  max_steps: 1000
  batch_size: 4
```

---

## Diffusion Config Example

```yaml
experiment:
  family: "diffusion"
  task: "image_generation"

model:
  type: "unet"
  noise_schedule: "cosine"

algorithm:
  objective: "denoising_score_matching"

data:
  dataset_name: "custom_images"

training:
  max_steps: 10000
  batch_size: 16
```

---

## RL Config Example

```yaml
experiment:
  family: "rl"
  task: "control"

model:
  type: "actor_critic"

algorithm:
  name: "ppo"
  gamma: 0.99
  gae_lambda: 0.95

environment:
  name: "CartPole-v1"

training:
  total_timesteps: 100000
```

---

## TAP Run Fields

Each TAP run should eventually store:

```text
experiment_family
task_type
algorithm_name
repo_name
config_path
launch_command
status
metrics
artifacts
```

Example for SLM:

```text
experiment_family: slm
task_type: causal_language_modeling
algorithm_name: transformer_lm
```

Example for RL:

```text
experiment_family: rl
task_type: control
algorithm_name: ppo
```

Example for diffusion:

```text
experiment_family: diffusion
task_type: image_generation
algorithm_name: ddpm
```

---

## Do Not Make TAP a Universal Neural Network Builder Too Early

Avoid making TAP combine arbitrary components like:

```text
SLM attention + diffusion sampler + PPO loss + random tokenizer
```

That will become messy and hard to validate.

Instead, TAP should combine components only through valid registries.

---

## Family-Specific Registries

### SLM Registry

```text
attention:
  - baseline
  - gqa
  - swa
  - xsa

norm:
  - rmsnorm
  - layernorm

mlp:
  - swiglu
  - gelu

tokenizer:
  - bpe
  - sentencepiece
```

### RL Registry

```text
algorithm:
  - dqn
  - ppo
  - sac

environment:
  - gym
  - custom

policy:
  - mlp
  - cnn
  - transformer
```

### Diffusion Registry

```text
model:
  - unet
  - dit

sampler:
  - ddpm
  - ddim

schedule:
  - linear
  - cosine
```

Each family should define its own valid components and compatibility rules.

---

## How to Combine Algorithms Safely

Think of combination as pipelines, not random mixing.

Good combinations:

```text
train SLM → evaluate SLM
train diffusion model → sample images → evaluate FID
train RL agent → evaluate policy
train vision encoder → use in RL policy
train tokenizer → use in SLM training
```

Future combinations could include:

```text
SLM planner → RL environment
diffusion world model → RL policy
vision encoder → diffusion conditioning
```

But TAP should represent these as:

```text
Run A produces artifact
Run B consumes artifact
```

Not as one giant mixed config.

---

## Future TAP Data Model

TAP should eventually support:

```text
Run
Artifact
ExperimentGroup
Pipeline
Dependency
```

Example:

```text
Run 1: train tokenizer
Artifact: tokenizer.json

Run 2: train SLM
Consumes: tokenizer.json
Produces: checkpoint.pt

Run 3: evaluate SLM
Consumes: checkpoint.pt
Produces: eval_metrics.json
```

This structure works for SLM, RL, diffusion, CV, and multimodal experiments.

---

## Standard Repo Commands for TAP

Every repo should eventually support these commands:

```bash
# Validate config
python -m package.main --config_path path.yaml --validate_only

# Run training
python -m package.main --config_path path.yaml

# Run evaluation
python -m package.eval --config_path path.yaml --checkpoint path.pt

# Print metadata
python -m package.main --config_path path.yaml --print_run_metadata
```

The SLM repo does not need all of these immediately, but this is the target convention.

---

## Standard Metrics Convention

TAP should store generic metrics, while allowing family-specific metrics.

### Common Metrics

```text
step
runtime_seconds
learning_rate
loss
train_loss
val_loss
samples_seen
tokens_seen
episodes_seen
created_at
source
```

### SLM-Specific Metrics

```text
tokens_per_second
perplexity
train_loss
val_loss
```

### Diffusion-Specific Metrics

```text
denoising_loss
fid
clip_score
sample_time
```

### RL-Specific Metrics

```text
episode_return
episode_length
policy_loss
value_loss
entropy
success_rate
```

TAP should not require every metric to exist.

Most metrics should be optional.

---

## Universal Metric JSONL Fallback

Every repo should eventually be able to write metrics to a local JSONL file.

Recommended path:

```text
outputs/runs/<run_id>/metrics.jsonl
```

Example line:

```json
{"step": 1, "train_loss": 6.2, "learning_rate": 0.0005, "tokens_seen": 512, "runtime_seconds": 3.1}
```

This gives TAP a universal fallback even without W&B.

---

## Artifact Convention

Every run should write an artifact manifest.

Recommended path:

```text
outputs/runs/<run_id>/artifacts.json
```

Example:

```json
{
  "run_id": "slm-smoke-001",
  "artifacts": [
    {
      "type": "checkpoint",
      "name": "latest",
      "path": "checkpoints/latest.pt"
    },
    {
      "type": "tokenizer",
      "name": "bpe-tokenizer",
      "path": "artifacts/tokenizer/tokenizer.json"
    },
    {
      "type": "config",
      "name": "config-snapshot",
      "path": "outputs/runs/slm-smoke-001/config.yaml"
    }
  ]
}
```

This allows TAP to track what each run produced.

---

## Smoke Test Convention

Every repo should have a local smoke test config.

Examples:

```text
configs/train/local_smoke.yaml
configs/train/smoke.yaml
```

The smoke test should prove:

```text
config loads
data loads
model builds
forward pass works
loss computes
optimizer step works
metrics log
artifacts save
```

The smoke test should avoid heavy dependencies like:

```text
S3
Ray
large datasets
long Slurm jobs
cloud services
```

---

## Git Hygiene Convention

Never commit:

```text
.env
.env.local
logs/
artifacts/
checkpoints/
wandb/
__pycache__/
*.out
*.err
*.pt
*.ckpt
```

Use:

```text
.env.example
```

for placeholder environment variables.

---

## Recommended Future Work for SLM Repo

Add these gradually:

```text
1. --validate_only mode
2. standard experiment section in configs
3. standard tracking section
4. artifact manifest output
5. metric JSONL output
6. TAP reads metric JSONL or W&B
```

Most important near-term addition:

```text
outputs/runs/<run_id>/metrics.jsonl
```

This helps TAP support real metrics without depending only on W&B.

---

## Final Rule

One repo can have its own internal logic, but it must expose a standard experiment interface.

That means every repo should share:

```text
same config shape
same launch pattern
same metrics logging shape
same artifact manifest shape
same smoke test pattern
same git hygiene
```

Then TAP can support:

```text
SLM
diffusion
RL
CV
multimodal
```

without becoming impossible to maintain.