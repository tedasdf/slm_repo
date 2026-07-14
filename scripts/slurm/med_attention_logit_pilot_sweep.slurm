#!/bin/bash
#SBATCH --job-name=med-attn-logit-1k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# 1k-step attention-logit multiplier pilot sweep for configs/train/ALG.yaml.
#
# Model is fixed:
#   L=8, H=8, d_model=256, d_ff=1024, seq_len=128, batch_size=32, fp32
#   qk_norm=false
#
# This runs alpha values sequentially in one Slurm allocation, matching the
# med_lr_sweep.slurm style and avoiding torchrun rendezvous port collisions.
#
# Usage:
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/med_attention_logit_pilot_sweep.slurm

set -eo pipefail

ALPHAS=(1.0 1.5 2.0 3.0 4.0)
LABELS=("attn-logit-alpha-1.0" "attn-logit-alpha-1.5" "attn-logit-alpha-2.0" "attn-logit-alpha-3.0" "attn-logit-alpha-4.0")

for IDX in "${!ALPHAS[@]}"; do
    ALPHA=${ALPHAS[$IDX]}
    ALPHA_LABEL=${LABELS[$IDX]}

    export CONFIG_PATH="configs/train/ALG.yaml"
    export CONFIG_OVERRIDES_JSON="{\"model\":{\"attention\":{\"attention_logit_multiplier\":${ALPHA}}},\"scheduler\":{\"warmup_steps\":300,\"t_max\":1000,\"eta_min\":0.0},\"trainer\":{\"max_steps\":1000,\"train_log_every\":50,\"eval_every\":1000,\"log_attention_diagnostics\":true,\"attention_diagnostic_layers\":[0,3,7]},\"logging\":{\"wandb_run_name\":null,\"wandb_tags\":[\"14m\",\"attention-logit-pilot\",\"${ALPHA_LABEL}\",\"alpha-${ALPHA}\",\"qk-norm-false\",\"1kstep\",\"attention-diagnostics\"]}}"

    echo "Sweep idx: $IDX"
    echo "Alpha:     $ALPHA"
    echo "Label:     $ALPHA_LABEL"
    echo "Config:    $CONFIG_PATH"
    echo "Overrides: $CONFIG_OVERRIDES_JSON"

    bash scripts/slurm/train.slurm
done
