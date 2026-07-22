#!/usr/bin/env sh
set -eu

# Follow-up training matrix for released-data experiments.
# The matrix is centered on the strongest current setting:
#   Dual SegFormer + ConvNeXt-Tiny + chv3 split ([0,1,2] / [3,4,5,6]).
# This script only runs experiments; it does not generate or edit paper text.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_experiments/chv3}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/chv3_summary.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
ENCODER="${ENCODER:-tu-convnext_tiny}"
PRETRAIN="${PRETRAIN:-true}"
EPOCHS="${EPOCHS:-100}"
INPUT_SIZE="${INPUT_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-cosine}"
AUG_PROB="${AUG_PROB:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-42}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
NORMALIZATION="${NORMALIZATION:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-false}"
DETERMINISTIC="${DETERMINISTIC:-true}"
PRIMARY_METRIC="${PRIMARY_METRIC:-miou}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
MIN_DELTA="${MIN_DELTA:-0.0}"
RUN_FUSION="${RUN_FUSION:-1}"
RUN_SIZE="${RUN_SIZE:-0}"
RUN_NOAUG="${RUN_NOAUG:-0}"
SIZE_INPUT="${SIZE_INPUT:-256}"
SIZE_BATCH_SIZE="${SIZE_BATCH_SIZE:-8}"

run_exp() {
  exp_prefix="$1"
  fusion="$2"
  loss="$3"
  augmentation="$4"
  mosaic_prob="$5"
  input_size="$6"
  batch_size="$7"

  for seed in ${SEEDS}; do
    exp_name="${exp_prefix}_seed${seed}"
    echo
    echo "=== ${exp_name}: chv3 ${fusion}, loss=${loss}, aug=${augmentation}, mosaic=${mosaic_prob}, size=${input_size} ==="

    if [ -n "${DATA_ROOT:-}" ]; then
      "${PYTHON_BIN}" train_ablation.py \
        --data-root "${DATA_ROOT}" \
        --train-split "${TRAIN_SPLIT}" \
        --val-split "${VAL_SPLIT}" \
        --output-dir "${OUTPUT_DIR}" \
        --summary-csv "${SUMMARY_CSV}" \
        --model-name auto \
        --arch dual_segformer \
        --encoder "${ENCODER}" \
        --pretrain "${PRETRAIN}" \
        --channels1 "0,1,2" \
        --channels2 "3,4,5,6" \
        --fusion "${fusion}" \
        --epochs "${EPOCHS}" \
        --input-size "${input_size}" \
        --batch-size "${batch_size}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --optimizer "${OPTIMIZER}" \
        --scheduler "${SCHEDULER}" \
        --loss "${loss}" \
        --augmentation "${augmentation}" \
        --aug-prob "${AUG_PROB}" \
        --mosaic-prob "${mosaic_prob}" \
        --num-workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        --seed "${seed}" \
        --val-interval "${VAL_INTERVAL}" \
        --save-interval "${SAVE_INTERVAL}" \
        --normalization "${NORMALIZATION}" \
        --mixed-precision "${MIXED_PRECISION}" \
        --deterministic "${DETERMINISTIC}" \
        --primary-metric "${PRIMARY_METRIC}" \
        --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
        --min-delta "${MIN_DELTA}" \
        --experiment-name "${exp_name}"
    else
      "${PYTHON_BIN}" train_ablation.py \
        --train-split "${TRAIN_SPLIT}" \
        --val-split "${VAL_SPLIT}" \
        --output-dir "${OUTPUT_DIR}" \
        --summary-csv "${SUMMARY_CSV}" \
        --model-name auto \
        --arch dual_segformer \
        --encoder "${ENCODER}" \
        --pretrain "${PRETRAIN}" \
        --channels1 "0,1,2" \
        --channels2 "3,4,5,6" \
        --fusion "${fusion}" \
        --epochs "${EPOCHS}" \
        --input-size "${input_size}" \
        --batch-size "${batch_size}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --optimizer "${OPTIMIZER}" \
        --scheduler "${SCHEDULER}" \
        --loss "${loss}" \
        --augmentation "${augmentation}" \
        --aug-prob "${AUG_PROB}" \
        --mosaic-prob "${mosaic_prob}" \
        --num-workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        --seed "${seed}" \
        --val-interval "${VAL_INTERVAL}" \
        --save-interval "${SAVE_INTERVAL}" \
        --normalization "${NORMALIZATION}" \
        --mixed-precision "${MIXED_PRECISION}" \
        --deterministic "${DETERMINISTIC}" \
        --primary-metric "${PRIMARY_METRIC}" \
        --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
        --min-delta "${MIN_DELTA}" \
        --experiment-name "${exp_name}"
    fi
  done
}

# Core matrix: test whether chv3 remains strong under loss and mosaic changes.
run_exp "exp_01_chv3_add_unetformer" "add" "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
run_exp "exp_02_chv3_add_dice" "add" "dice" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
run_exp "exp_03_chv3_add_ce" "add" "ce" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
run_exp "exp_04_chv3_add_no_mosaic" "add" "unetformer" "mars" "0.0" "${INPUT_SIZE}" "${BATCH_SIZE}"

if [ "${RUN_FUSION}" = "1" ]; then
  run_exp "exp_05_chv3_cat_unetformer" "cat" "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
  run_exp "exp_06_chv3_att_unetformer" "att" "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
  run_exp "exp_07_chv3_moe_unetformer" "moe" "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
  run_exp "exp_08_chv3_moev2_unetformer" "moev2" "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
else
  echo
  echo "Skipping chv3 fusion variants because RUN_FUSION=${RUN_FUSION}."
fi

if [ "${RUN_SIZE}" = "1" ]; then
  run_exp "exp_09_chv3_add_size${SIZE_INPUT}" "add" "unetformer" "mars" "0.5" "${SIZE_INPUT}" "${SIZE_BATCH_SIZE}"
else
  echo
  echo "Skipping input-size follow-up because RUN_SIZE=${RUN_SIZE}."
fi

if [ "${RUN_NOAUG}" = "1" ]; then
  run_exp "exp_10_chv3_add_no_aug_no_mosaic" "add" "unetformer" "none" "0.0" "${INPUT_SIZE}" "${BATCH_SIZE}"
else
  echo
  echo "Skipping no-augmentation follow-up because RUN_NOAUG=${RUN_NOAUG}."
fi

echo
echo "Chv3 training finished. Results are under ${OUTPUT_DIR}."
echo "Summary CSV: ${SUMMARY_CSV}"
