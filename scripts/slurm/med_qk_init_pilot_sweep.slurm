#!/bin/bash
#SBATCH --job-name=med-qk-init-1k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# 1k-step Q/K init multiplier pilot sweep for configs/train/ALG.yaml.
#
# qk_gain_init multiplies only q_proj/k_proj init std. Since both Q and K are
# scaled by m, initial attention-score scale should move roughly like m^2.
#
# Usage:
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/med_qk_init_pilot_sweep.slurm

set -eo pipefail

MULTS=(1.0 1.224744871 1.414213562 1.732050808 2.0)
EFFECTIVE_SCALES=(1.0 1.5 2.0 3.0 4.0)
LABELS=("qk-init-m-1.0" "qk-init-m-1.2247" "qk-init-m-1.4142" "qk-init-m-1.7321" "qk-init-m-2.0")

for IDX in "${!MULTS[@]}"; do
    MULT=${MULTS[$IDX]}
    EFFECTIVE_SCALE=${EFFECTIVE_SCALES[$IDX]}
    MULT_LABEL=${LABELS[$IDX]}

    export CONFIG_PATH="configs/train/ALG.yaml"
    export CONFIG_OVERRIDES_JSON="{\"model\":{\"attention\":{\"qk_gain_init\":${MULT},\"attention_logit_multiplier\":1.0}},\"scheduler\":{\"warmup_steps\":300,\"t_max\":1000,\"eta_min\":0.0},\"trainer\":{\"max_steps\":1000,\"train_log_every\":50,\"eval_every\":1000,\"log_attention_diagnostics\":true,\"attention_diagnostic_layers\":[0,3,7]},\"logging\":{\"wandb_run_name\":null,\"wandb_tags\":[\"14m\",\"qk-init-pilot\",\"${MULT_LABEL}\",\"qk-init-m-${MULT}\",\"expected-score-scale-${EFFECTIVE_SCALE}\",\"qk-norm-false\",\"1kstep\",\"attention-diagnostics\"]}}"

    echo "Sweep idx:              $IDX"
    echo "QK init multiplier:     $MULT"
    echo "Expected score scale:   $EFFECTIVE_SCALE"
    echo "Label:                  $MULT_LABEL"
    echo "Config:                 $CONFIG_PATH"
    echo "Overrides:              $CONFIG_OVERRIDES_JSON"

    bash scripts/slurm/train.slurm
done
