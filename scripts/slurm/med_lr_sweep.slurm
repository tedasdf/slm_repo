#!/bin/bash
#SBATCH --job-name=med-lr-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

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
#
# This runs the LR values sequentially in one Slurm allocation. That avoids
# torchrun rendezvous port collisions between concurrent array tasks.

set -eo pipefail

LRS=(3e-2 5e-2 1e-1 2e-1 3e-1)
LABELS=("high" "higher" "very high" "near divergent" "divergent")

LABEL="grad norm inspection 14M"

for IDX in "${!LRS[@]}"; do
    LR=${LRS[$IDX]}
    LR_LABEL=${LABELS[$IDX]}

    export CONFIG_PATH="configs/train/med_lr.yaml"
    export CONFIG_OVERRIDES_JSON="{\"optimizer\":{\"lr\":${LR}},\"logging\":{\"wandb_run_name\":null,\"wandb_tags\":[\"grad-norm-inspection\",\"14m\",\"med-lr\",\"lr-sweep\"]}}"

    echo "Sweep idx: $IDX"
    echo "LR:        $LR"
    echo "LR label:  $LR_LABEL"
    echo "Label:     $LABEL"
    echo "Config:    $CONFIG_PATH"
    echo "Overrides: $CONFIG_OVERRIDES_JSON"

    bash scripts/slurm/train.slurm
done
