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

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Started at: $(date)"

module load miniforge3
conda activate slm_ven

cd /home/slo/vf38_scratch2/sloo0021/slm_repo

mkdir -p logs/slurm

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Hugging Face cache on scratch, not home
export HF_HOME=/home/slo/vf38_scratch2/sloo0021/hf_cache
export HF_DATASETS_CACHE=/home/slo/vf38_scratch2/sloo0021/hf_cache/datasets
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

nvidia-smi

CONFIG_PATH="${CONFIG_PATH:-configs/train/fineweb_edu_smoke.yaml}"

echo "Config path: ${CONFIG_PATH}"

torchrun --nproc_per_node=1 -m src.slm.main --config_path "${CONFIG_PATH}"

echo "Finished at: $(date)"