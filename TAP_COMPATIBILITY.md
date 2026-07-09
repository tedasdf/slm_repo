# TAP Compatibility — Training Setup Snapshot

Current branch: `main` | Date: 2026-06-05

---

## Model Architecture

| Parameter | Value | Notes |
|---|---|---|
| Architecture | Decoder-only Transformer | Pre-norm, causal mask |
| `num_layers` | 8 | |
| `model_dim` (d) | 512 | |
| `num_heads` | 8 | |
| `head_dim` | 64 | d / num_heads |
| `num_kv_heads` | 8 | MHA (no GQA by default) |
| `vocab_size` | 32 000 | BPE tokenizer |
| `max_seq_len` | 512 | |
| MLP type | GELU | hidden = 4d = 2048 |
| Norm type | RMSNorm | eps = 1e-6, no bias |
| Positional encoding | RoPE | rope_base = 10 000 |
| Attention scaling | 1/√head_dim | via `scaled_dot_product_attention` |
| Weight tying | **False** | embedding ≠ lm_head |
| `bias=False` | Everywhere | all `nn.Linear`, norms |
| QK-norm | Off (toggleable) | `attention.qk_norm: true` to enable |

### Initialisation

| Component | Scheme |
|---|---|
| Embeddings | N(0, 1/√d) = N(0, 1/√512) ≈ N(0, 0.044) |
| All linear weights | Truncated-normal, std = 1/√fan\_in, truncated at ±2σ |
| Biases | n/a — no biases |

Controlled by `model.init.use_fan_in_init: true` (default).

---

## Optimiser

| Parameter | Value |
|---|---|
| Optimiser | AdamW |
| β₁ | 0.9 |
| β₂ | 0.95 |
| ε | 1e-8 |
| Weight decay | 1e-4 (independent) |
| Gradient clipping | Global norm 1.0 |

---

## LR Schedule

| Phase | Steps | Shape |
|---|---|---|
| Warmup | 0 → 5 000 | Linear, 0 → peak LR |
| Decay | 5 000 → 100 000 | Cosine → min LR |
| Peak LR | 3e-4 | |
| Min LR (floor) | 1e-5 | |

Implemented as `LinearLR` + `CosineAnnealingLR` via `SequentialLR`.

---

## Training

| Parameter | Value |
|---|---|
| Precision | bf16 (no fp16, no grad scaler) |
| Batch size | 32 sequences |
| Grad accum steps | 8 |
| **Effective batch** | **256 sequences × 512 tokens = 131 072 tokens/step** |
| Max steps | 10 000 (override per sweep) |
| Seed | 42 (torch + numpy + cuda + data order) |
| Loss | Autoregressive cross-entropy (next-token prediction) |
| Auxiliary z-loss | coeff = 1e-4 · E[logsumexp(logits)²] |
| Anomaly detection | Off |

---

## Data

| Parameter | Value |
|---|---|
| Dataset | `HuggingFaceFW/fineweb-edu`, config `sample-10BT` |
| Loading | HuggingFace streaming (no DVC, no Ray) |
| Val split | Synthetic — 0.5% hash-based split from train stream |
| Text field | `text` |
| Tokeniser | BPE, vocab 32 000, trained on 500 k samples before fit |
| Tokeniser path | `artifacts/tokenizer/baseline/tokenizer.json` (reuse if exists) |
| Shuffle buffer | 10 000 |
| Sequence packing | Token loader: contiguous mmap chunks (no padding). Text/streaming path: truncates long docs, pads short docs with -100 mask (no loss corruption; strict packing requires pre-tokenised bin mode) |
| HF revision pinning | `data.hf_revision` field available — set to pin a commit |

---

## Toggleable Experimental Knobs

| Knob | Config key | Default | Notes |
|---|---|---|---|
| QK-norm | `attention.qk_norm` | `false` | RMSNorm(head_dim) on Q and K per head |
| GQA | `attention.num_kv_heads` | = num_heads | Set < num_heads for GQA |
| Sliding window attn | `attention.window_size` | null | Requires `attention_type: swa` |
| Exclusion self-attn (XSA) | `attention.attention_type` | `baseline` | Custom variant |
| Fan-in init | `model.init.use_fan_in_init` | `true` | False → fixed std=0.02 |
| Logit softcap | `model.logit_softcap` | null | Gemma-style tanh cap |
| Checkpoint resume | `trainer.resume_from_checkpoint` | null | Path to `last.pt` |
| Compile | `trainer.compile_model` | `false` | `torch.compile` |

---

## Checkpoint Behaviour

| File | When saved |
|---|---|
| `artifacts/checkpoints/step_N.pt` | Every `checkpoint_every=1000` steps |
| `artifacts/checkpoints/last.pt` | Same cadence, always overwritten — Slurm resume target |
| `artifacts/checkpoints/best.pt` | When val_loss improves |

Checkpoint contains: model weights, optimizer state, scheduler state, grad scaler state, full `TrainState` (step, tokens seen, best val loss).

---

## What Is NOT in the Repo

| Item | Status |
|---|---|
| Evaluation harness (lm-eval / perplexity benchmarks) | Not implemented |
| Hessian / spectral analysis | Not implemented (README TODO) |
| Multi-node DDP | Single-node only; NCCL wiring exists |
| Packed streaming text | Deferred — use token mode for strict packing |
| Scaling law sweep | Implemented (`experiments/scaling_law.py`) but config not updated to fineweb-edu yet |
