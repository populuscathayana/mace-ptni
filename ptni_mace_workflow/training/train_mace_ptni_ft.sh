#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash ptni_mace_workflow/training/train_mace_ptni_ft.sh [--workspace DIR] [--dataset NAME] [--run-name NAME] [--epochs N] [--patience N]

Legacy positional form is also accepted:
  bash ptni_mace_workflow/training/train_mace_ptni_ft.sh DATA_DIR RUN_NAME

Environment:
  MACE_WORKSPACE, DEVICE, SEED, FOUNDATION_MODEL, BATCH_SIZE,
  VALID_BATCH_SIZE, MAX_NUM_EPOCHS, PATIENCE, SAVE_ALL_CHECKPOINTS,
  RESTART_LATEST, ALLOW_EXISTING_RUN, WANDB, WANDB_PROJECT, WANDB_NAME
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
DATASET="ptni_split"
DATA_DIR=""
RUN_NAME="ptni_binary_mace_ft"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
FOUNDATION_MODEL="${FOUNDATION_MODEL:-medium-mpa-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-${BATCH_SIZE}}"
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-80}"
PATIENCE="${PATIENCE:-20}"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE_ARG="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --run-name|--name) RUN_NAME="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --foundation-model) FOUNDATION_MODEL="$2"; shift 2 ;;
    --max-num-epochs|--epochs) MAX_NUM_EPOCHS="$2"; shift 2 ;;
    --patience|--early-stop-patience) PATIENCE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  DATA_DIR="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then
  RUN_NAME="${POSITIONAL[1]}"
fi

cd "${REPO_ROOT}"
mkdir -p "${WORKSPACE_ARG}"
WORKSPACE="$(cd "${WORKSPACE_ARG}" && pwd)"

if [[ -z "${DATA_DIR}" ]]; then
  DATA_DIR="${WORKSPACE}/datasets/${DATASET}"
elif [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${REPO_ROOT}/${DATA_DIR}"
fi

TRAIN_FILE="${DATA_DIR}/train.extxyz"
VALID_FILE="${DATA_DIR}/valid.extxyz"
TEST_FILE="${DATA_DIR}/test.extxyz"
RUN_DIR="${WORKSPACE}/runs/training/${RUN_NAME}"

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VALID_FILE}" || ! -f "${TEST_FILE}" ]]; then
  echo "Missing split files under ${DATA_DIR}." >&2
  echo "Expected train.extxyz, valid.extxyz, test.extxyz." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/results" "${RUN_DIR}/models" "${RUN_DIR}/wandb"

shopt -s nullglob
EXISTING_CHECKPOINTS=("${RUN_DIR}/checkpoints/${RUN_NAME}"_run-*_epoch-*.pt)
shopt -u nullglob
if [[ "${RESTART_LATEST:-0}" != "1" && "${ALLOW_EXISTING_RUN:-0}" != "1" && ${#EXISTING_CHECKPOINTS[@]} -gt 0 ]]; then
  echo "Existing checkpoints found for run ${RUN_NAME}, but RESTART_LATEST=1 was not passed." >&2
  echo "Use RESTART_LATEST=1 to resume, choose a new RUN_NAME, or set ALLOW_EXISTING_RUN=1 if intentional." >&2
  printf 'First checkpoint seen: %s\n' "${EXISTING_CHECKPOINTS[0]}" >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${SAVE_ALL_CHECKPOINTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_all_checkpoints --keep_checkpoints)
fi
if [[ "${RESTART_LATEST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--restart_latest)
fi
if [[ "${WANDB:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb)
  EXTRA_ARGS+=(--wandb_project "${WANDB_PROJECT:-ptni-mace}")
  EXTRA_ARGS+=(--wandb_name "${WANDB_NAME:-${RUN_NAME}}")
  EXTRA_ARGS+=(--wandb_dir wandb)
fi

cat > "${RUN_DIR}/run_manifest.json" <<EOF
{
  "kind": "training",
  "mode": "fine_tune",
  "run_name": "${RUN_NAME}",
  "workspace": "${WORKSPACE}",
  "data_dir": "${DATA_DIR}",
  "train_file": "${TRAIN_FILE}",
  "valid_file": "${VALID_FILE}",
  "test_file": "${TEST_FILE}",
  "device": "${DEVICE}",
  "seed": "${SEED}",
  "foundation_model": "${FOUNDATION_MODEL}",
  "batch_size": "${BATCH_SIZE}",
  "valid_batch_size": "${VALID_BATCH_SIZE}",
  "max_num_epochs": "${MAX_NUM_EPOCHS}",
  "patience": "${PATIENCE}",
  "save_all_checkpoints": "${SAVE_ALL_CHECKPOINTS:-0}",
  "restart_latest": "${RESTART_LATEST:-0}",
  "wandb": "${WANDB:-0}"
}
EOF

cat <<SETTINGS
MACE fine-tune launch settings:
  WORKSPACE=${WORKSPACE}
  RUN_DIR=${RUN_DIR}
  DATA_DIR=${DATA_DIR}
  RUN_NAME=${RUN_NAME}
  DEVICE=${DEVICE}
  SEED=${SEED}
  SAVE_ALL_CHECKPOINTS=${SAVE_ALL_CHECKPOINTS:-0}
  RESTART_LATEST=${RESTART_LATEST:-0}
  WANDB=${WANDB:-0}
  MAX_NUM_EPOCHS=${MAX_NUM_EPOCHS}
  PATIENCE=${PATIENCE}
SETTINGS

cd "${RUN_DIR}"
mace_run_train \
  --name "${RUN_NAME}" \
  --seed "${SEED}" \
  --train_file "${TRAIN_FILE}" \
  --valid_file "${VALID_FILE}" \
  --test_file "${TEST_FILE}" \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --foundation_model "${FOUNDATION_MODEL}" \
  --E0s average \
  --batch_size "${BATCH_SIZE}" \
  --valid_batch_size "${VALID_BATCH_SIZE}" \
  --max_num_epochs "${MAX_NUM_EPOCHS}" \
  --patience "${PATIENCE}" \
  --energy_weight 1.0 \
  --forces_weight 10.0 \
  --config_type_weights '{"bulk":1.0,"slab":1.0,"neb":2.0,"Default":1.0}' \
  --default_dtype float64 \
  --device "${DEVICE}" \
  --checkpoints_dir checkpoints \
  --log_dir logs \
  --results_dir results \
  --model_dir models \
  "${EXTRA_ARGS[@]}"
