#!/bin/bash
#SBATCH --job-name=tiny-lr-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

# LR sweep for configs/train/tiny_lr.yaml.
#
# Model is fixed:
#   L=4, H=4, d_model=128, d_ff=512, seq_len=128, batch_size=32, fp32
#
# Usage:
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/tiny_lr_sweep.slurm
#   # limit concurrent GPUs:
#   sbatch --array=0-1%2 scripts/slurm/tiny_lr_sweep.slurm

set -eo pipefail

# Full LR sweep:
# LRS=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1)
LRS=(3e-1 1e0)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

export CONFIG_PATH="configs/train/tiny_lr.yaml"
export CONFIG_OVERRIDES_JSON="{\"optimizer\":{\"lr\":${LR}},\"logging\":{\"wandb_tags\":[\"tiny-lr\",\"no-warmup\",\"high-lr-test\",\"fineweb-edu\"]}}"

echo "Array idx: $SLURM_ARRAY_TASK_ID"
echo "LR:        $LR"
echo "Config:    $CONFIG_PATH"
echo "Overrides: $CONFIG_OVERRIDES_JSON"

bash scripts/slurm/train.slurm
