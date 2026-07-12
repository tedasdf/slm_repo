#!/bin/bash
#SBATCH --job-name=attn-logit-cal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -eo pipefail

export SCRATCH="${SCRATCH:-/home/slo/vf38_scratch2/sloo0021}"

mkdir -p logs/slurm

module load miniforge3
conda activate slm_ven

cd /home/slo/vf38_scratch2/sloo0021/slm_repo

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

python scripts/calibrate_attention_logits.py \
    --config_path "${CONFIG_PATH:-configs/train/ALG.yaml}" \
    --overrides "${CONFIG_OVERRIDES_JSON:-{}}" \
    --batch-source "${BATCH_SOURCE:-synthetic}" \
    --layers "${LAYERS:-0,3,7}" \
    --alphas "${ALPHAS:-1,1.25,1.5,2,3,4}" \
    --device cuda
