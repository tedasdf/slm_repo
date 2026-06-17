#!/bin/bash
#SBATCH --job-name=lr-instability-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-27
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

# Depth/width x learning-rate instability sweep — 4 model sizes x 7 LRs = 28 tasks.
# Base config: configs/train/figure2_qk_false_repro.yaml, everything else
# overridden per task below (same CONFIG_PATH/CONFIG_OVERRIDES_JSON mechanism
# as train.slurm).
#
# Sweep grid:
#   model:  (L=3,d=256,h=4)=2.4M   (L=3,d=512,h=8)=9.4M
#           (L=6,d=512,h=8)=19M    (L=6,d=768,h=12)=42M
#   lr:     1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1
#
# 10000 steps each. Schedule matches figure2_qk_false_repro.yaml shape:
#   warmup_steps=5000, t_max=95000 (cosine decays slowly — LR stays near
#   peak through 10k steps, giving a fair instability test per model/lr).
# effective batch: 256x512=131072 tokens/step (batch_size=256, grad_accum=1).
#
# Estimated GPU-hours per task @ 10000 steps (L40S, mfu=0.35, pure FLOPs):
#   2.4M=0.40h  9.4M=0.95h  19M=1.33h  42M=2.48h  (~41 GPU-h total, 28 tasks)
# Actual wall-clock is higher — enable_anomaly_detection=true (inherited)
# adds ~1.5-2x backward overhead (42M: ~5h step-compute alone).
# --time=08:00:00 covers the 42M model with data/eval/overhead headroom.
#
# Usage:
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/lr_instability_sweep.slurm
#   # limit concurrent GPUs:  sbatch --array=0-27%4 scripts/slurm/lr_instability_sweep.slurm

set -eo pipefail
# Note: -u (unbound variable check) is intentionally omitted — module/conda
# env scripts reference $SCRATCH and other M3-specific vars without exporting them.

export SCRATCH="${SCRATCH:-/home/slo/vf38_scratch2/sloo0021}"

mkdir -p logs/slurm

# ── sweep grid ──────────────────────────────────────────────────────────────
NAMES=(2p4M 9p4M 19M 42M)
LAYERS=(3 3 6 6)
DIMS=(256 512 512 768)
HEADS=(4 8 8 12)

LRS=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1)

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / ${#LRS[@]} ))
LR_IDX=$(( SLURM_ARRAY_TASK_ID % ${#LRS[@]} ))

NAME=${NAMES[$CONFIG_IDX]}
NUM_LAYERS=${LAYERS[$CONFIG_IDX]}
MODEL_DIM=${DIMS[$CONFIG_IDX]}
NUM_HEADS=${HEADS[$CONFIG_IDX]}
LR=${LRS[$LR_IDX]}

# figure2_qk_false_repro.yaml leaves mlp.hidden_dim unset (derived as
# mlp_mult * model_dim at load time), but OmegaConf.structured(RunConfig)
# bakes that derivation in for the *default* model_dim=512 as a concrete
# hidden_dim=2048 in the schema. Overriding model_dim alone would not
# re-derive it, so set hidden_dim explicitly (mlp_mult=4.0, per figure2).
HIDDEN_DIM=$(( MODEL_DIM * 4 ))

RUN_NAME="lr-sweep-${NAME}-lr${LR}"

export CONFIG_PATH="configs/train/figure2_qk_false_repro.yaml"
export CONFIG_OVERRIDES_JSON="{\"model\":{\"num_layers\":${NUM_LAYERS},\"model_dim\":${MODEL_DIM},\"attention\":{\"num_heads\":${NUM_HEADS},\"num_kv_heads\":${NUM_HEADS}},\"mlp\":{\"hidden_dim\":${HIDDEN_DIM}}},\"optimizer\":{\"lr\":${LR}},\"scheduler\":{\"warmup_steps\":5000,\"t_max\":95000},\"trainer\":{\"max_steps\":10000,\"grad_accum_steps\":8},\"data\":{\"batch_size\":32},\"logging\":{\"wandb_run_name\":\"${RUN_NAME}\",\"wandb_tags\":[\"lr-sweep\",\"qk-false\",\"c4\"]}}"
export TAP_RUN_ID="$RUN_NAME"

echo "Job ID:    $SLURM_JOB_ID"
echo "Array idx: $SLURM_ARRAY_TASK_ID  ->  model=$NAME (L=$NUM_LAYERS d=$MODEL_DIM h=$NUM_HEADS), lr=$LR"
echo "Host:      $(hostname)"
echo "Started:   $(date)"
echo "Config:    $CONFIG_PATH"
echo "Overrides: $CONFIG_OVERRIDES_JSON"
echo "Run ID:    $TAP_RUN_ID"

# ── conda ─────────────────────────────────────────────────────────────────────
module load miniforge3
# conda activate is a shell function — needs conda init sourced first
# shellcheck source=/dev/null
conda activate slm_ven

# ── repo ──────────────────────────────────────────────────────────────────────
cd /home/slo/vf38_scratch2/sloo0021/slm_repo

# ── threading ─────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# ── HF auth ───────────────────────────────────────────────────────────────────
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ ! -f "$HF_TOKEN_FILE" ]]; then
    echo "ERROR: HF token not found at $HF_TOKEN_FILE" >&2
    exit 1
fi
export HF_TOKEN
HF_TOKEN=$(cat "$HF_TOKEN_FILE")
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$SCRATCH/.cache/hf_datasets"
mkdir -p "$HF_DATASETS_CACHE"

# ── GPU check ─────────────────────────────────────────────────────────────────
nvidia-smi

# ── run ───────────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=1 -m src.slm.main \
    --config_path "$CONFIG_PATH" \
    --overrides   "$CONFIG_OVERRIDES_JSON"

echo "Finished: $(date)"
