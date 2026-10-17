#!/usr/bin/env sh
# Shared environment resolution for the group launcher scripts in scripts/.
#
# Usage (from a launcher that has already cd'd to the repository root):
#   . "${SCRIPT_DIR}/_env.sh"
#
# Every variable below can be overridden by exporting it before the launcher:
#   DATA_ROOT  MARS_DATA_ROOT  MAPPED_DATA_ROOT  DEVICE  SEEDS  BATCH_SIZE
#   NUM_WORKERS  PYTHON_BIN  CONFIG  GROUP  EXPERIMENTS  FINALIZE_RESULTS
#   DRY_RUN  SKIP_MODEL_INIT  STRICT_DETERMINISM  RUN_ROOT  EXPECTED_RUNS
#
# When DATA_ROOT is left unset it is taken from the same variables the trainer
# itself reads (MMLSV2_DATA_ROOT / MARS_DATA_ROOT / DATA_ROOT, including .env via
# env_utils), so launchers do not need a baked-in path.

DATA_ROOT="${DATA_ROOT:-${MARS_DATA_ROOT:-${MAPPED_DATA_ROOT:-${MMLSV2_DATA_ROOT:-}}}}"
export DATA_ROOT
if [ -n "${DATA_ROOT}" ]; then
  export MARS_DATA_ROOT="${DATA_ROOT}"
  export MMLSV2_DATA_ROOT="${DATA_ROOT}"
fi

DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-41 42 43}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/mmlsv2_paper_protocol.json}"
BATCH_SIZE="${BATCH_SIZE:-}"
EXPERIMENTS="${EXPERIMENTS:-}"
# Training results stay in the project output tree by default. Result copying
# to the dataset filesystem is an explicit opt-in for legacy workflows only.
FINALIZE_RESULTS="${FINALIZE_RESULTS:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MODEL_INIT="${SKIP_MODEL_INIT:-0}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-1}"

export PYTHON_BIN CONFIG DEVICE SEEDS NUM_WORKERS STRICT_DETERMINISM FINALIZE_RESULTS
export PYTHONUNBUFFERED=1

if [ -n "${BATCH_SIZE}" ]; then
  export BATCH_SIZE
fi
if [ -n "${EXPERIMENTS}" ]; then
  export EXPERIMENTS
fi
if [ "${DRY_RUN}" != "0" ]; then
  export DRY_RUN
fi
if [ "${SKIP_MODEL_INIT}" != "0" ]; then
  export SKIP_MODEL_INIT
fi
