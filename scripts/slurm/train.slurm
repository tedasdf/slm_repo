#!/bin/bash
#SBATCH --job-name=slm-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Usage:
#   mkdir -p logs/slurm   # must exist before sbatch (SLURM opens the log before the script runs)
#   sbatch scripts/slurm/train.slurm
#
# Override config or hyperparams at submission time:
#   CONFIG_PATH=configs/train/instability_probe.yaml sbatch scripts/slurm/train.slurm
#   CONFIG_PATH=configs/train/baseline.yaml \
#   CONFIG_OVERRIDES_JSON='{"trainer":{"lr":3e-3,"max_steps":2000}}' \
#   TAP_RUN_ID=run_001 \
#   sbatch scripts/slurm/train.slurm

set -eo pipefail
# Note: -u (unbound variable check) is intentionally omitted — module/conda
# env scripts reference $SCRATCH and other M3-specific vars without exporting them.

export SCRATCH="${SCRATCH:-/home/slo/vf38_scratch2/sloo0021}"

mkdir -p logs/slurm

# Normalise CONFIG_OVERRIDES_JSON before use.
# Avoid ${VAR:-{}} — when VAR is set to '{}', bash emits the value then a
# literal '}' from the expansion syntax, producing '{}}' which breaks json.loads.
_OVERRIDES="${CONFIG_OVERRIDES_JSON:-}"
[[ -z "$_OVERRIDES" ]] && _OVERRIDES='{}'

echo "Job ID:    $SLURM_JOB_ID"
echo "Host:      $(hostname)"
echo "Started:   $(date)"
echo "Config:    ${CONFIG_PATH:-configs/train/baseline.yaml}"
echo "Overrides: $_OVERRIDES"
echo "Run ID:    ${TAP_RUN_ID:-<not set>}"

# Give each Slurm job its own torchrun rendezvous port. Without this,
# concurrent array tasks on the same node all try the default 29500.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-0} % 40000) ))}"
echo "Master:    $MASTER_ADDR:$MASTER_PORT"

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
torchrun --nproc_per_node=1 \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    -m src.slm.main \
    --config_path "${CONFIG_PATH:-configs/train/baseline.yaml}" \
    --overrides   "$_OVERRIDES"

echo "Finished: $(date)"
