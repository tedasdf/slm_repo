#!/bin/bash
#SBATCH --job-name=med-attn-logit-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# 5-step smoke test for the attention-logit multiplier pipeline.
#
# Usage:
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/med_attention_logit_smoke_sweep.slurm

set -eo pipefail

ALPHAS=(1.0 4.0)
LABELS=("attn-logit-alpha-1.0" "attn-logit-alpha-4.0")

for IDX in "${!ALPHAS[@]}"; do
    ALPHA=${ALPHAS[$IDX]}
    ALPHA_LABEL=${LABELS[$IDX]}

    export CONFIG_PATH="configs/train/ALG.yaml"
    export CONFIG_OVERRIDES_JSON="{\"model\":{\"attention\":{\"attention_logit_multiplier\":${ALPHA}}},\"scheduler\":{\"warmup_steps\":1,\"t_max\":5,\"eta_min\":0.0},\"trainer\":{\"max_steps\":5,\"train_log_every\":1,\"eval_every\":5,\"log_attention_diagnostics\":true,\"attention_diagnostic_layers\":[0,3,7]},\"logging\":{\"wandb_run_name\":null,\"wandb_tags\":[\"14m\",\"attention-logit-smoke\",\"${ALPHA_LABEL}\",\"alpha-${ALPHA}\",\"qk-norm-false\",\"5step\",\"attention-diagnostics\"]}}"

    echo "Sweep idx: $IDX"
    echo "Alpha:     $ALPHA"
    echo "Label:     $ALPHA_LABEL"
    echo "Config:    $CONFIG_PATH"
    echo "Overrides: $CONFIG_OVERRIDES_JSON"

    bash scripts/slurm/train.slurm
done
