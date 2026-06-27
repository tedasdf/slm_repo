#!/bin/bash
#SBATCH --job-name=scaling-law
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started at: $(date)"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-0} % 40000) ))}"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

module load miniforge3
conda activate slm_ven

cd /home/slo/vf38_scratch2/sloo0021/slm_repo

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

nvidia-smi

torchrun --nproc_per_node=4 \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    -m src.slm.main --config_path configs/train/smoke.yaml

echo "Finished at: $(date)"
