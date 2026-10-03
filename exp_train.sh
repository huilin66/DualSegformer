#!/usr/bin/env bash
set -euo pipefail

# Experiment entry point. The default is intentionally conservative:
# reproduce the historical competition model first. Other groups are started
# explicitly so that a new-model run cannot be mixed with an unfrozen baseline.
#
# Usage:
#   bash exp_train.sh                         # competition reproduction
#   bash exp_train.sh reproduce               # same as the default
#   bash exp_train.sh fusion                  # legacy fusion ablations
#   bash exp_train.sh capacity                # backbone capacity comparison
#   bash exp_train.sh new_model               # SKG/new-model pipeline
#   bash exp_train.sh test                    # evaluate a selected best checkpoint on public test
#   bash exp_train.sh all                     # run groups in the planned order
#
# Common overrides:
#   MARS_DATA_ROOT=/path/to/mapped DATA_ROOT=/path/to/mapped bash exp_train.sh reproduce
#   SEEDS="42 123 7" bash exp_train.sh capacity
#   PYTHON_BIN=/path/to/python DEVICE=cuda:1 bash exp_train.sh reproduce

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash exp_train.sh [reproduce|fusion|capacity|new_model|test|smoke|all]

Groups:
  reproduce  Reproduce the historical competition configuration.
  fusion     Compare legacy add/cat/attention/MoE fusion variants.
  capacity   Compare Tiny/Small/Base/Large backbones.
  new_model  Run the SKG/new-model pipeline after the baseline is frozen.
  test       Evaluate a selected checkpoint on the labeled public test split.
  smoke      Run the repository's data and training smoke tests.
  all        Run groups in the order reproduce -> fusion -> capacity -> new_model.

The default group is reproduce. Configure paths, seeds, device, and epochs
through environment variables documented in the individual group scripts.
EOF
}

run_group() {
  group_script="$1"
  echo ""
  echo "######## ${group_script} ########"
  bash "${REPO_ROOT}/scripts/${group_script}"
}

GROUP="${1:-reproduce}"
if [ "$#" -gt 1 ]; then
  usage >&2
  exit 2
fi

case "${GROUP}" in
  reproduce|baseline)
    run_group "exp01_reproduce_competition.sh"
    ;;
  fusion|ablation)
    run_group "exp02_legacy_fusion_ablation.sh"
    ;;
  capacity|backbone)
    run_group "exp03_legacy_capacity.sh"
    ;;
  new_model|skg)
    run_group "exp04_skg_new_model.sh"
    ;;
  test|evaluate)
    run_group "evaluate_reproduction_test.sh"
    ;;
  smoke)
    echo "######## smoke tests ########"
    bash "${REPO_ROOT}/scripts/smoke_test_data.sh"
    bash "${REPO_ROOT}/scripts/smoke_test_train.sh"
    ;;
  all)
    run_group "exp01_reproduce_competition.sh"
    run_group "exp02_legacy_fusion_ablation.sh"
    run_group "exp03_legacy_capacity.sh"
    run_group "exp04_skg_new_model.sh"
    ;;
  help|-h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown experiment group: ${GROUP}" >&2
    usage >&2
    exit 2
    ;;
esac

echo ""
echo "Experiment group completed: ${GROUP}"
