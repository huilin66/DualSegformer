#!/usr/bin/env sh
set -eu

# Group: all -- core + ablations + extended, sequentially.
# 10 core baselines (M1-M4, M6-M11) + 9 ablations (A1 A2 A5 B2 B3 C2 L1 L2 L3)
# = 19 experiments x 3 seeds = 57 formal runs.
#
# Sequential on purpose: the paper protocol is deterministic and these runs are
# meant to share one GPU. For a one-shot process use GROUP=all instead:
#   GROUP=all bash scripts/train_paper_experiments.sh
#
# Overrides are the same as the individual launchers; EXPERIMENTS is honoured
# inside each group that contains the requested ids (ids from another group are
# rejected by the runner, so do not mix groups in one EXPERIMENTS value).

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

echo "######## group=all seeds=${SEEDS} batch=${BATCH_SIZE:-protocol-default} ########"

for group_script in run_core.sh run_ablations.sh run_extended.sh; do
  echo "######## launching ${group_script} ########"
  sh "${SCRIPT_DIR}/${group_script}"
done

echo "######## group=all completed ########"
