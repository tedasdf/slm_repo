#!/bin/bash
#SBATCH --job-name=med-attn-lr-10k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# 10k-step attention-only learning-rate multiplier sweep.
#
# Usage:
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/med_attention_lr_sweep.slurm

set -eo pipefail

MULTS=(1.0 1.5 2.0 3.0 4.0)
LABELS=("attn-lr-mult-1.0" "attn-lr-mult-1.5" "attn-lr-mult-2.0" "attn-lr-mult-3.0" "attn-lr-mult-4.0")

for IDX in "${!MULTS[@]}"; do
    MULT=${MULTS[$IDX]}
    MULT_LABEL=${LABELS[$IDX]}

    export CONFIG_PATH="configs/train/ALG.yaml"
    export CONFIG_OVERRIDES_JSON="{\"optimizer\":{\"attention_lr_multiplier\":${MULT}},\"scheduler\":{\"warmup_steps\":300,\"t_max\":10000,\"eta_min\":0.0},\"trainer\":{\"max_steps\":10000,\"train_log_every\":50,\"eval_every\":200,\"log_attention_diagnostics\":true,\"attention_diagnostic_layers\":[0,3,7]},\"logging\":{\"wandb_run_name\":null,\"wandb_tags\":[\"14m\",\"attention-lr-sweep\",\"${MULT_LABEL}\",\"attn-lr-mult-${MULT}\",\"qk-norm-false\",\"10kstep\",\"attention-diagnostics\"]}}"

    echo "Sweep idx:           $IDX"
    echo "Attention LR mult:   $MULT"
    echo "Label:               $MULT_LABEL"
    echo "Config:              $CONFIG_PATH"
    echo "Overrides:           $CONFIG_OVERRIDES_JSON"

    bash scripts/slurm/train.slurm
done
