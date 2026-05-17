#!/bin/bash
#SBATCH --job-name=tap-m1-success
#SBATCH --output=logs/slurm/tap-m1-success-%j.out
#SBATCH --error=logs/slurm/tap-m1-success-%j.err
#SBATCH --time=00:02:00

echo "TAP M1 success smoke test"
sleep 10
exit 0
