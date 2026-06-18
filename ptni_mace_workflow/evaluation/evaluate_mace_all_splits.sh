#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?Usage: evaluate_mace_all_splits.sh MODEL_PATH [DATA_DIR] [DEVICE] [OUT_DIR]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
cd "${REPO_ROOT}"
mkdir -p "${WORKSPACE_ARG}"
WORKSPACE="$(cd "${WORKSPACE_ARG}" && pwd)"

DATA_DIR="${2:-${WORKSPACE}/datasets/ptni_split}"
DEVICE="${3:-cuda}"
OUT_DIR="${4:-${WORKSPACE}/runs/evaluation/manual_batched/ptni_split/predictions}"
SPLITS="${SPLITS:-train valid test}"

mkdir -p "${OUT_DIR}"

if [[ "${MODEL_PATH}" == *.pt ]]; then
  cat >&2 <<'EOF'
The model path ends with .pt. MACE training .pt files are usually restart
checkpoints, not directly evaluable model files. mace_eval_configs expects a
serialized MACE model, usually ending in .model.

Use a .model file from the training output if available, or export/finish the
checkpoint first. Example:
  find . -name '*.model' -o -name '*.pt'
EOF
  exit 2
fi

if command -v mace_eval_configs >/dev/null 2>&1; then
  EVAL_CMD=(mace_eval_configs)
else
  echo "mace_eval_configs not found in PATH; falling back to python -m mace.cli.eval_configs"
  EVAL_CMD=(python -m mace.cli.eval_configs)
fi

for SPLIT in ${SPLITS}; do
  CONFIGS="${DATA_DIR}/${SPLIT}.extxyz"
  OUTPUT="${OUT_DIR}/${SPLIT}_pred.extxyz"
  if [[ ! -f "${CONFIGS}" ]]; then
    echo "Missing ${CONFIGS}" >&2
    exit 1
  fi
  echo "Evaluating ${SPLIT}: ${CONFIGS}"
  "${EVAL_CMD[@]}" \
    --configs "${CONFIGS}" \
    --model "${MODEL_PATH}" \
    --output "${OUTPUT}" \
    --device "${DEVICE}"
done

echo "Prediction files written to ${OUT_DIR}"
echo "Now score them with:"
echo "python ${SCRIPT_DIR}/score_mace_predictions_extxyz.py \\"
for SPLIT in ${SPLITS}; do
  echo "  --pred ${SPLIT}=${OUT_DIR}/${SPLIT}_pred.extxyz \\"
done
echo "  --out-csv ${OUT_DIR}/split_metrics.csv \\"
echo "  --out-md ${OUT_DIR}/split_metrics.md"
