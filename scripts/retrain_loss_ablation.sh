#!/usr/bin/env sh
set -eu

# Re-train the loss ablation family only:
#   L1 = CE-only      (paper-facing M5)
#   L2 = Dice-only
#   L3 = CE + Dice    (legacy combined-loss comparison)
#
# The default keeps completed runs in the project-side output tree and does
# not copy checkpoints to the dataset output directory during training. This
# avoids making a large checkpoint write to a flaky shared filesystem part of
# the training critical path. Set FINALIZE_RESULTS=1 only when that sync is
# explicitly desired.
#
# Examples:
#   DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls \
#     bash scripts/retrain_loss_ablation.sh
#   DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls SEEDS="41" \
#     DRY_RUN=1 bash scripts/retrain_loss_ablation.sh
#   DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls FINALIZE_RESULTS=1 \
#     bash scripts/retrain_loss_ablation.sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

GROUP="ablations"
EXPERIMENTS="L1,L2,L3"
SEEDS="${SEEDS:-41 42 43}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FINALIZE_RESULTS="${FINALIZE_RESULTS:-0}"
FRESH="${FRESH:-1}"
ARCHIVE_DATASET_OLD="${ARCHIVE_DATASET_OLD:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_experiments/pmdfnet_mmlsv2}"

export GROUP EXPERIMENTS SEEDS DEVICE BATCH_SIZE NUM_WORKERS FINALIZE_RESULTS

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -n "${DATA_ROOT}" ] || die "DATA_ROOT 未设置（也可用 MARS_DATA_ROOT / MAPPED_DATA_ROOT / MMLSV2_DATA_ROOT）"
[ -d "${DATA_ROOT}/train/images" ] || die "数据集缺少 train/images: ${DATA_ROOT}"
[ -d "${DATA_ROOT}/val/masks" ] || die "数据集缺少 val/masks: ${DATA_ROOT}"

say "=============================================================="
say " L1 / L2 / L3 loss ablation retraining"
say "   config       : ${CONFIG}"
say "   data         : ${DATA_ROOT}"
say "   experiments  : ${EXPERIMENTS}"
say "   seeds        : ${SEEDS}"
say "   device       : ${DEVICE}"
say "   batch/workers: ${BATCH_SIZE}/${NUM_WORKERS}"
say "   output       : ${OUTPUT_ROOT}"
say "   finalize     : ${FINALIZE_RESULTS}"
say "   fresh        : ${FRESH}"
say "=============================================================="

# Keep the project-side formal run tree clean. The move is recoverable and is
# deliberately limited to L1/L2/L3 directories; it never deletes data.
if [ "${FRESH}" = "1" ] && [ "${DRY_RUN:-0}" = "0" ]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  project_backup="${REPO_ROOT}/outputs_experiments/_backup_before_loss_retrain_${stamp}/pmdfnet_mmlsv2"
  mkdir -p "${project_backup}"
  moved=0
  for pattern in 'L1_*' 'L2_*' 'L3_*'; do
    for run_dir in "${OUTPUT_ROOT}/${pattern}"; do
      [ -d "${run_dir}" ] || continue
      mv -- "${run_dir}" "${project_backup}/"
      moved=$((moved + 1))
    done
  done
  say "[fresh] archived ${moved} old project-side L1/L2/L3 runs to ${project_backup}"

  # Dataset-side archival is opt-in because it touches the shared output
  # filesystem. It is still a move, never an overwrite or deletion.
  if [ "${ARCHIVE_DATASET_OLD}" = "1" ] && [ -d "${DATA_ROOT}/outputs" ]; then
    dataset_backup="${DATA_ROOT}/backup_before_loss_retrain_${stamp}/outputs"
    mkdir -p "${dataset_backup}"
    moved_dataset=0
    for pattern in 'L1_*' 'L2_*' 'L3_*'; do
      for run_dir in "${DATA_ROOT}/outputs/${pattern}"; do
        [ -d "${run_dir}" ] || continue
        mv -- "${run_dir}" "${dataset_backup}/"
        moved_dataset=$((moved_dataset + 1))
      done
    done
    say "[fresh] archived ${moved_dataset} old dataset-side L1/L2/L3 runs to ${dataset_backup}"
  fi
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  say "[dry-run] configuration and data checks passed; no training started"
  exit 0
fi

say "[train] starting L1/L2/L3 for seeds: ${SEEDS}"
# train_paper_experiments.sh writes project-side runs under the protocol's
# output_dir. FINALIZE_RESULTS=0 keeps the training path independent of the
# dataset output filesystem.
sh "${SCRIPT_DIR}/train_paper_experiments.sh"

say "[verify] checking completed project-side runs"
"${PYTHON_BIN}" - "${REPO_ROOT}/${OUTPUT_ROOT}" "${SEEDS}" <<'PY'
import glob
import os
import sys

root = sys.argv[1]
seeds = sys.argv[2].split()
expected = []
for seed in seeds:
    for prefix in ("L1_ce_only", "L2_dice_only", "L3_ce_dice"):
        expected.append((prefix, seed))

failed = False
for prefix, seed in expected:
    matches = sorted(glob.glob(os.path.join(root, f"{prefix}_s{seed}_*")))
    if not matches:
        print(f"FAIL missing: {prefix}_s{seed}")
        failed = True
        continue
    run_dir = matches[-1]
    log_path = os.path.join(run_dir, "logs", "train.log")
    best_path = os.path.join(run_dir, "checkpoints", "best.pth")
    try:
        log_text = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        log_text = ""
    ok = "Training completed" in log_text and os.path.isfile(best_path)
    print(f"{'OK' if ok else 'FAIL'} {os.path.basename(run_dir)}")
    failed = failed or not ok

if failed:
    raise SystemExit(1)
PY

say "[done] all requested L1/L2/L3 runs passed the project-side completeness check"
say "       project output: ${REPO_ROOT}/${OUTPUT_ROOT}"
if [ "${FINALIZE_RESULTS}" = "0" ]; then
  say "       dataset sync: disabled (set FINALIZE_RESULTS=1 for a future run)"
fi
