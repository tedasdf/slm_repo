%%bash
set -euo pipefail

DATA_DIR="/content/dolma_v1_6_sample"
PARALLEL_DOWNLOADS=8
DOLMA_VERSION="v1_6-sample"

if [ ! -d "dolma" ]; then
  git clone https://huggingface.co/datasets/allenai/dolma
fi

mkdir -p "${DATA_DIR}"

cat "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "${DATA_DIR}"

echo "Download complete."
find "${DATA_DIR}" | head -20