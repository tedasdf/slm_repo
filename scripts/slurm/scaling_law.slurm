#!/bin/bash
#SBATCH --job-name=scaling-law
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started at: $(date)"

module load miniforge3
conda activate slm_ven

cd /home/slo/vf38_scratch2/sloo0021/slm_repo

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

nvidia-smi

# python -m src.slm.main --config_path configs/train/baseline.yaml --experiment_path configs/experiment/scaling_law.yaml --experiment

torchrun --nproc_per_node=4 -m src.slm.main --config_path configs/train/smoke.yaml

echo "Finished at: $(date)"