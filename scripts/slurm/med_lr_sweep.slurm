#!/bin/bash
#SBATCH --job-name=med-lr-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

# LR sweep for configs/train/med_lr.yaml.
#
# Model is fixed:
#   L=8, H=8, d_model=256, d_ff=1024, seq_len=128, batch_size=32, fp32
#   total params = 14,492,160
#   core params  =  6,300,160  (excl. embedding + head)
#
# Scheduler:
#   constant LR, no warmup
#
# Usage:
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/med_lr_sweep.slurm
#   # limit concurrent GPUs:
#   sbatch --array=0-6%2 scripts/slurm/med_lr_sweep.slurm

set -eo pipefail

LRS=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

LABEL="med-lr-no-warmup-total-14492160-core-6300160"

export CONFIG_PATH="configs/train/med_lr.yaml"
export CONFIG_OVERRIDES_JSON="{\"optimizer\":{\"lr\":${LR}},\"logging\":{\"wandb_tags\":[\"${LABEL}\",\"med-lr\",\"no-warmup\",\"lr-sweep\",\"fineweb-edu\",\"total-params-14492160\",\"core-params-6300160\"]}}"

echo "Array idx: $SLURM_ARRAY_TASK_ID"
echo "LR:        $LR"
echo "Label:     $LABEL"
echo "Config:    $CONFIG_PATH"
echo "Overrides: $CONFIG_OVERRIDES_JSON"

bash scripts/slurm/train.slurm
