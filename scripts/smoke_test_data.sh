#!/usr/bin/env sh
set -eu

# Smoke test 1: verify CLI parsing, .env data root, dataset import, and one sample read.
# This does not initialize a model or run training.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_jstar/smoke/data}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/smoke_data_summary.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
INPUT_SIZE="${INPUT_SIZE:-64}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"

echo "=== Smoke test: Python syntax ==="
"${PYTHON_BIN}" -m py_compile train_ablation.py env_utils.py

echo
echo "=== Smoke test: train_ablation.py --help ==="
"${PYTHON_BIN}" train_ablation.py --help >/dev/null

echo
echo "=== Smoke test: dataset dry-run ==="
if [ -n "${DATA_ROOT:-}" ]; then
  "${PYTHON_BIN}" train_ablation.py \
    --data-root "${DATA_ROOT}" \
    --experiment-name smoke_data \
    --output-dir "${OUTPUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --input-size "${INPUT_SIZE}" \
    --batch-size 1 \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --num-workers "${NUM_WORKERS}" \
    --dry-run \
    --skip-model-init
else
  "${PYTHON_BIN}" train_ablation.py \
    --experiment-name smoke_data \
    --output-dir "${OUTPUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --input-size "${INPUT_SIZE}" \
    --batch-size 1 \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --num-workers "${NUM_WORKERS}" \
    --dry-run \
    --skip-model-init
fi

echo
echo "Smoke data test passed."
echo "Output dir: ${OUTPUT_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
