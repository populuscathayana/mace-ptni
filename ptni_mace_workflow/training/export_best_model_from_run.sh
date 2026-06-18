#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash ptni_mace_workflow/training/export_best_model_from_run.sh \
    --workspace mace_workspace \
    --run-name ptni_binary_mace_ft \
    --model-tag ft_best_loss

Options:
  --workspace DIR       Runtime workspace. Default: MACE_WORKSPACE or mace_workspace.
  --run-name NAME       Training run name under workspace/runs/training/<NAME>.
  --model-tag TAG       Destination under workspace/models/<TAG>/model.model.
  --metric loss|energy|force
  --epoch N             Export a specific epoch instead of parsing the log.
  --run-id ID           MACE run id/seed in log filenames. Default: 123.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
RUN_NAME=""
MODEL_TAG=""
METRIC="loss"
EPOCH=""
RUN_ID="${RUN_ID:-123}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE_ARG="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --metric) METRIC="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${RUN_NAME}" || -z "${MODEL_TAG}" ]]; then
  echo "--run-name and --model-tag are required." >&2
  usage >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${WORKSPACE_ARG}"
WORKSPACE="$(cd "${WORKSPACE_ARG}" && pwd)"
RUN_DIR="${WORKSPACE}/runs/training/${RUN_NAME}"
LOG_PATH="${RUN_DIR}/logs/${RUN_NAME}_run-${RUN_ID}.log"
CHECKPOINTS_DIR="${RUN_DIR}/checkpoints"
OUT_MODEL="${WORKSPACE}/models/${MODEL_TAG}/model.model"

TEMPLATE_MODEL=""
for candidate in \
  "${RUN_DIR}/models/${RUN_NAME}_run-${RUN_ID}.model" \
  "${RUN_DIR}/checkpoints/${RUN_NAME}_run-${RUN_ID}.model" \
  "${RUN_DIR}/${RUN_NAME}_run-${RUN_ID}.model"
do
  if [[ -f "${candidate}" ]]; then
    TEMPLATE_MODEL="${candidate}"
    break
  fi
done

if [[ ! -f "${LOG_PATH}" ]]; then
  echo "Log not found: ${LOG_PATH}" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
  echo "Checkpoints directory not found: ${CHECKPOINTS_DIR}" >&2
  exit 2
fi
if [[ -z "${TEMPLATE_MODEL}" ]]; then
  echo "Template model not found. Looked in models/, checkpoints/, and run root under ${RUN_DIR}." >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT_MODEL}")"
EPOCH_ARGS=()
if [[ -n "${EPOCH}" ]]; then
  EPOCH_ARGS=(--epoch "${EPOCH}")
fi

python "${SCRIPT_DIR}/export_best_mace_checkpoint_from_log.py" \
  --log "${LOG_PATH}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --template-model "${TEMPLATE_MODEL}" \
  --run-name "${RUN_NAME}" \
  --metric "${METRIC}" \
  --output-model "${OUT_MODEL}" \
  "${EPOCH_ARGS[@]}"

cat > "${WORKSPACE}/models/${MODEL_TAG}/model_manifest.json" <<EOF
{
  "model_tag": "${MODEL_TAG}",
  "source_run": "${RUN_NAME}",
  "metric": "${METRIC}",
  "epoch": "${EPOCH}",
  "run_dir": "${RUN_DIR}",
  "log": "${LOG_PATH}",
  "checkpoints_dir": "${CHECKPOINTS_DIR}",
  "template_model": "${TEMPLATE_MODEL}",
  "model": "${OUT_MODEL}"
}
EOF

echo "Exported model: ${OUT_MODEL}"
