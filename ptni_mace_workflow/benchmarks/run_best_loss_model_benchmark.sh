#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
SUITE="${SUITE:-lattice,strained_neb,pt111_pes}"
DEVICE="${DEVICE:-cuda}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-float64}"

MODEL_TAGS="${MODEL_TAGS:-ft_best_loss scratch_best_loss}"

echo "Compatibility wrapper: running run_benchmark_suite.sh for MODEL_TAGS=${MODEL_TAGS}"
for tag in ${MODEL_TAGS}; do
  bash "${SCRIPT_DIR}/run_benchmark_suite.sh" \
    --workspace "${WORKSPACE_ARG}" \
    --model-tag "${tag}" \
    --suite "${SUITE}" \
    --device "${DEVICE}" \
    --default-dtype "${DEFAULT_DTYPE}"
done
