#!/bin/bash
#SBATCH --job-name=med-attn-input-10k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# 10k-step normalized attention-input multiplier sweep.
#
# Usage:
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/med_attention_input_sweep.slurm

set -eo pipefail

MULTS=(1.0 1.224744871 1.414213562 1.732050808 2.0)
EFFECTIVE_SCALES=(1.0 1.5 2.0 3.0 4.0)
LABELS=("attn-input-c-1.0" "attn-input-c-1.2247" "attn-input-c-1.4142" "attn-input-c-1.7321" "attn-input-c-2.0")

for IDX in "${!MULTS[@]}"; do
    MULT=${MULTS[$IDX]}
    EFFECTIVE_SCALE=${EFFECTIVE_SCALES[$IDX]}
    MULT_LABEL=${LABELS[$IDX]}

    export CONFIG_PATH="configs/train/ALG.yaml"
    export CONFIG_OVERRIDES_JSON="{\"model\":{\"attention\":{\"attention_input_multiplier\":${MULT},\"attention_logit_multiplier\":1.0}},\"scheduler\":{\"warmup_steps\":300,\"t_max\":10000,\"eta_min\":0.0},\"trainer\":{\"max_steps\":10000,\"train_log_every\":50,\"eval_every\":200,\"log_attention_diagnostics\":true,\"attention_diagnostic_layers\":[0,3,7]},\"logging\":{\"wandb_run_name\":null,\"wandb_tags\":[\"14m\",\"attention-input-sweep\",\"${MULT_LABEL}\",\"attn-input-c-${MULT}\",\"expected-score-scale-${EFFECTIVE_SCALE}\",\"qk-norm-false\",\"10kstep\",\"attention-diagnostics\"]}}"

    echo "Sweep idx:              $IDX"
    echo "Attention input mult:   $MULT"
    echo "Expected score scale:   $EFFECTIVE_SCALE"
    echo "Label:                  $MULT_LABEL"
    echo "Config:                 $CONFIG_PATH"
    echo "Overrides:              $CONFIG_OVERRIDES_JSON"

    bash scripts/slurm/train.slurm
done
