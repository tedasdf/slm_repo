#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/data/tokenizer/base.yaml}"
PYTHON_BIN="${PYTHON_BIN:-./.slm_wsl/bin/python}"

OUTPUT="$($PYTHON_BIN -m src.slm.data.tokenization --config_path "$CONFIG_PATH")"

echo "$OUTPUT"

RUN_DIRS="$(echo "$OUTPUT" | grep '^RUN_DIR=' | cut -d= -f2-)"

if [[ -z "$RUN_DIRS" ]]; then
  echo "Error: no RUN_DIR values were returned by tokenization"
  exit 1
fi

while IFS= read -r run_dir; do
  [[ -z "$run_dir" ]] && continue

  echo "Adding to DVC: $run_dir"
  dvc add "$run_dir"

  DVC_FILE="${run_dir%/}.dvc"
  if [[ ! -f "$DVC_FILE" ]]; then
    echo "Error: expected DVC file not found: $DVC_FILE"
    exit 1
  fi

  echo "Adding Git metadata: $DVC_FILE"
  git add "$DVC_FILE" .gitignore

  echo "Pushing data to DVC remote: $DVC_FILE"
  dvc push "$DVC_FILE"

  echo "Done: $run_dir"
done <<< "$RUN_DIRS"