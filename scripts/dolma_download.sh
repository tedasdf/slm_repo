DATA_DIR="artifacts/data/dolma_v1_6_sample"
PARALLEL_DOWNLOADS=8
DOLMA_VERSION="v1_6-sample"

mkdir -p "${DATA_DIR}"

tr -d '\r' < "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -P "${DATA_DIR}"