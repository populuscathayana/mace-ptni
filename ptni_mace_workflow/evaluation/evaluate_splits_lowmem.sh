#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash ptni_mace_workflow/evaluation/evaluate_splits_lowmem.sh \
    --workspace mace_workspace \
    --model-tag ft_best_loss \
    --dataset ptni_split \
    --device cuda

Options:
  --workspace DIR       Runtime workspace. Default: MACE_WORKSPACE or mace_workspace.
  --model-tag TAG       Model tag under workspace/models/<TAG>/model.model.
  --model PATH          Explicit .model path. Overrides --model-tag path lookup.
  --dataset NAME        Dataset under workspace/datasets/<NAME>. Default: ptni_split.
  --data-dir DIR        Explicit split directory. Overrides --dataset path lookup.
  --out-dir DIR         Explicit output directory.
  --device DEVICE       cuda or cpu. Default: cuda.
  --default-dtype TYPE  Default dtype passed to MACECalculator. Default: float64.
  --splits "..."        Space-separated split names. Default: train valid test.
  --limit N             Evaluate at most N configs per split.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
MODEL_TAG=""
MODEL_PATH=""
DATASET="ptni_split"
DATA_DIR=""
OUT_DIR=""
DEVICE="${DEVICE:-cuda}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-float64}"
SPLITS="${SPLITS:-train valid test}"
LIMIT=""
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-10}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE_ARG="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --default-dtype) DEFAULT_DTYPE="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "${REPO_ROOT}"
mkdir -p "${WORKSPACE_ARG}"
WORKSPACE="$(cd "${WORKSPACE_ARG}" && pwd)"

if [[ -z "${MODEL_TAG}" && -z "${MODEL_PATH}" ]]; then
  echo "Either --model-tag or --model is required." >&2
  exit 2
fi
if [[ -z "${MODEL_TAG}" ]]; then
  MODEL_TAG="$(basename "${MODEL_PATH%.*}")"
fi
if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="${WORKSPACE}/models/${MODEL_TAG}/model.model"
elif [[ "${MODEL_PATH}" != /* ]]; then
  MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}"
fi
if [[ "${MODEL_PATH}" == *.pt ]]; then
  echo "Refusing .pt checkpoint for evaluation: ${MODEL_PATH}" >&2
  echo "Export it to .model first." >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found: ${MODEL_PATH}" >&2
  exit 2
fi

if [[ -z "${DATA_DIR}" ]]; then
  DATA_DIR="${WORKSPACE}/datasets/${DATASET}"
elif [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${REPO_ROOT}/${DATA_DIR}"
fi
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${WORKSPACE}/runs/evaluation/${MODEL_TAG}/${DATASET}/lowmem"
elif [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi

mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/run_manifest.csv"
echo "split,configs,model,prediction,device,default_dtype,limit" > "${MANIFEST}"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

echo "Low-memory split evaluation:"
echo "  WORKSPACE=${WORKSPACE}"
echo "  MODEL_TAG=${MODEL_TAG}"
echo "  MODEL=${MODEL_PATH}"
echo "  DATA_DIR=${DATA_DIR}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  SPLITS=${SPLITS}"
echo "  DEVICE=${DEVICE}"

PRED_ARGS=()
for SPLIT in ${SPLITS}; do
  CONFIGS="${DATA_DIR}/${SPLIT}.extxyz"
  OUTPUT="${OUT_DIR}/${SPLIT}_pred.extxyz"
  if [[ ! -f "${CONFIGS}" ]]; then
    echo "Missing split file: ${CONFIGS}" >&2
    exit 1
  fi
  echo
  echo "=== ${SPLIT}: one-by-one prediction ==="
  python "${SCRIPT_DIR}/evaluate_mace_extxyz_one_by_one.py" \
    --configs "${CONFIGS}" \
    --model "${MODEL_PATH}" \
    --output "${OUTPUT}" \
    --device "${DEVICE}" \
    --default-dtype "${DEFAULT_DTYPE}" \
    --empty-cache-every "${EMPTY_CACHE_EVERY}" \
    "${LIMIT_ARGS[@]}"
  PRED_ARGS+=(--pred "${SPLIT}=${OUTPUT}")
  echo "${SPLIT},${CONFIGS},${MODEL_PATH},${OUTPUT},${DEVICE},${DEFAULT_DTYPE},${LIMIT}" >> "${MANIFEST}"
done

echo
echo "=== scoring splits ==="
python "${SCRIPT_DIR}/score_mace_predictions_extxyz.py" \
  "${PRED_ARGS[@]}" \
  --out-csv "${OUT_DIR}/split_metrics.csv" \
  --out-md "${OUT_DIR}/split_metrics.md"

echo "Evaluation complete: ${OUT_DIR}"
