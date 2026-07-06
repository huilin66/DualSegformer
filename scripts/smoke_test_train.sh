#!/usr/bin/env sh
set -eu

# Smoke test 2: run a tiny one-epoch training/evaluation loop.
# It uses the generic arch/encoder path with pretrain disabled to avoid network downloads.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_jstar/smoke/train}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/smoke_train_summary.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
ARCH="${ARCH:-segformer}"
ENCODER="${ENCODER:-tu-convnext_tiny}"
LOSS="${LOSS:-ce}"
INPUT_SIZE="${INPUT_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-4}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-4}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-none}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

echo "=== Smoke test: tiny train/eval loop ==="
if [ -n "${DATA_ROOT:-}" ]; then
  "${PYTHON_BIN}" train_ablation.py \
    --data-root "${DATA_ROOT}" \
    --experiment-name smoke_train \
    --output-dir "${OUTPUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --model-name "" \
    --arch "${ARCH}" \
    --encoder "${ENCODER}" \
    --pretrain false \
    --loss "${LOSS}" \
    --epochs 1 \
    --input-size "${INPUT_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --optimizer "${OPTIMIZER}" \
    --scheduler "${SCHEDULER}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --mixed-precision false \
    --deterministic true
else
  "${PYTHON_BIN}" train_ablation.py \
    --experiment-name smoke_train \
    --output-dir "${OUTPUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --model-name "" \
    --arch "${ARCH}" \
    --encoder "${ENCODER}" \
    --pretrain false \
    --loss "${LOSS}" \
    --epochs 1 \
    --input-size "${INPUT_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --optimizer "${OPTIMIZER}" \
    --scheduler "${SCHEDULER}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --mixed-precision false \
    --deterministic true
fi

echo
echo "Smoke train test passed."
echo "Output dir: ${OUTPUT_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
