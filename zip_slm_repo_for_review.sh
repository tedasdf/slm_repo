#!/usr/bin/env bash
set -euo pipefail

OUT_FILE="../slm_repo_for_review_$(date +%Y%m%d_%H%M%S).zip"

echo "Creating $OUT_FILE from current slm_repo..."

zip -r "$OUT_FILE" . \
  -x ".slm_Wsl/*"\
  -x ".dvc/*" \
  -x ".git/*" \
  -x ".env" \
  -x ".env.*" \
  -x "logs/*" \
  -x "*.egg-info/*" \
  -x "src/*.egg-info/*" \
  -x "*.out" \
  -x "*.err" \
  -x "__pycache__/*" \
  -x "**/__pycache__/*" \
  -x ".venv/*" \
  -x ".dev_venv/*" \
  -x "venv/*" \
  -x "env/*" \
  -x "wandb/*" \
  -x "artifacts/*" \
  -x "data/*" \
  -x "checkpoints/*" \
  -x "runs/*" \
  -x "outputs/*" \
  -x ".mypy_cache/*" \
  -x ".pytest_cache/*" \
  -x ".ruff_cache/*" \
  -x "*.pt" \
  -x "*.pth" \
  -x "*.bin" \
  -x "*.safetensors" \
  -x "**/*.pt" \
  -x "**/*.pth" \
  -x "**/*.bin" \
  -x "**/*.safetensors" \
  -x "**/*.pkl" \
  -x "**/*.npy" \
  -x "**/*.npz" \
  -x "**/*.parquet" \
  -x "**/*.jsonl" \
  -x "**/*.jsonl.gz" \
  -x "**/*.log"

if [ -f .env.example ]; then
  zip -g "$OUT_FILE" .env.example >/dev/null
fi

echo "Done: $OUT_FILE"
ls -lh "$OUT_FILE"
