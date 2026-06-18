#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?Usage: evaluate_mace_ptni.sh MODEL_PATH [DATA_DIR] [DEVICE]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
cd "${REPO_ROOT}"
mkdir -p "${WORKSPACE_ARG}"
WORKSPACE="$(cd "${WORKSPACE_ARG}" && pwd)"

DATA_DIR="${2:-${WORKSPACE}/datasets/ptni_split}"
DEVICE="${3:-cuda}"
OUT_DIR="${WORKSPACE}/runs/evaluation/manual_test/ptni_split"

mkdir -p "${OUT_DIR}"

mace_eval_configs \
  --configs "${DATA_DIR}/test.extxyz" \
  --model "${MODEL_PATH}" \
  --output "${OUT_DIR}/test_pred.extxyz" \
  --device "${DEVICE}"

echo "Prediction file: ${OUT_DIR}/test_pred.extxyz"
