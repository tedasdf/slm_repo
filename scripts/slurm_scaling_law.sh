#!/bin/bash
#SBATCH --job-name=scaling-law
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

echo "Job ID:    $SLURM_JOB_ID"
echo "Host:      $(hostname)"
echo "Started:   $(date)"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-0} % 40000) ))}"
echo "Master:    $MASTER_ADDR:$MASTER_PORT"

# ── conda ──────────────────────────────────────────────────────────────────────
# conda activate is a shell function defined by conda init — not available in
# batch scripts unless you source the init script explicitly first.
module load miniforge3
# shellcheck source=/dev/null
conda activate slm_ven

# ── repo ───────────────────────────────────────────────────────────────────────
cd /home/slo/vf38_scratch2/sloo0021/slm_repo

# ── threading ─────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# ── HF auth ───────────────────────────────────────────────────────────────────
# Token written once on the login node via:
#   echo "hf_YOUR_TOKEN" > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ ! -f "$HF_TOKEN_FILE" ]]; then
    echo "ERROR: HF token not found at $HF_TOKEN_FILE" >&2
    echo "Run on login node: echo 'hf_YOUR_TOKEN' > $HF_TOKEN_FILE && chmod 600 $HF_TOKEN_FILE" >&2
    exit 1
fi
export HF_TOKEN
HF_TOKEN=$(cat "$HF_TOKEN_FILE")
export HF_HOME="$HOME/.cache/huggingface"

# ── HF dataset cache → scratch (avoids filling home quota) ────────────────────
# $SCRATCH is not guaranteed on M3 — use the known scratch path instead.
SCRATCH_DIR="/home/slo/vf38_scratch2/sloo0021"
export HF_DATASETS_CACHE="$SCRATCH_DIR/.cache/hf_datasets"
mkdir -p "$HF_DATASETS_CACHE"

# ── GPU check ─────────────────────────────────────────────────────────────────
nvidia-smi

# ── run ───────────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=1 \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    -m src.slm.main \
    --config_path configs/train/smoke.yaml

echo "Finished: $(date)"
