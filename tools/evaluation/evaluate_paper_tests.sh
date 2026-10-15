#!/usr/bin/env bash
set -euo pipefail

# Evaluate the validation-selected checkpoint for all formal PMDFNet paper runs.
# Results are written to each project-side run directory. Dataset outputs are
# intentionally not touched; archive them separately after evaluation.

REPO_ROOT="$(CDPATH= cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_ROOT="${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}"

mapfile -t RUN_DIRS < <(
  find "$RUN_ROOT" -type f -path '*/logs/train.log' \
    -exec grep -l 'Training completed' {} + \
    | grep -E '/(A1|A2|A5|B2|B3|C2|L1|L2)_.*_s(41|42|43)_' \
    | sed 's#/logs/train.log##' \
    | sort
)

if [[ "${#RUN_DIRS[@]}" -ne 24 ]]; then
  printf 'Expected 24 formal paper runs, found %s.\n' "${#RUN_DIRS[@]}" >&2
  printf '%s\n' "${RUN_DIRS[@]}" >&2
  exit 1
fi

echo "[paper-test] data=$DATA_ROOT device=$DEVICE batch=$BATCH_SIZE workers=$NUM_WORKERS"
echo "[paper-test] formal_runs=${#RUN_DIRS[@]}"

for run_dir in "${RUN_DIRS[@]}"; do
  checkpoint="$run_dir/checkpoints/best.pth"
  run_config="$run_dir/config.json"
  if [[ ! -f "$checkpoint" || ! -f "$run_config" ]]; then
    echo "[paper-test] missing checkpoint/config: $run_dir" >&2
    exit 1
  fi

  if [[ -f "$run_dir/test_metrics.json" ]]; then
    echo "[paper-test] skip existing: $run_dir"
    continue
  fi

  echo "[paper-test] start: $run_dir"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" tools/evaluation/evaluate_test_metrics.py \
      --data-root "$DATA_ROOT" \
      --checkpoint "$checkpoint" \
      --run-config "$run_config" \
      --output-dir "$run_dir" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS"
  echo "[paper-test] done: $run_dir"
done

echo "[paper-test] all formal runs evaluated"
