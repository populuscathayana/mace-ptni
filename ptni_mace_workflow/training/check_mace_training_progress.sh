#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
RUN_NAME="${RUN_NAME:-ptni_binary_mace_ft}"
RUN_ID="${RUN_ID:-123}"

cd "${REPO_ROOT}"
mkdir -p "${WORKSPACE_ARG}"
WORKSPACE="$(cd "${WORKSPACE_ARG}" && pwd)"
RUN_DIR="${WORKSPACE}/runs/training/${RUN_NAME}"

LOG_PATH="${1:-${RUN_DIR}/logs/${RUN_NAME}_run-${RUN_ID}.log}"
JSONL_PATH="${2:-${RUN_DIR}/results/${RUN_NAME}_run-${RUN_ID}_train.txt}"
CHECKPOINTS_DIR="${3:-${RUN_DIR}/checkpoints}"
OUT_DIR="${4:-${RUN_DIR}/training_checks}"

python "${SCRIPT_DIR}/check_mace_training_progress.py" \
  --log "${LOG_PATH}" \
  --jsonl "${JSONL_PATH}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --out-dir "${OUT_DIR}" \
  --plot
