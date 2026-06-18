#!/usr/bin/env bash
set -euo pipefail

# Run low-memory MACE single-point predictions on the NP benchmark package and
# score both structure-level errors and 00/01/02 NEB triplet barriers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
cd "${REPO_ROOT}"
mkdir -p "${WORKSPACE_ARG}"
WORKSPACE="$(cd "${WORKSPACE_ARG}" && pwd)"

PACKAGE_DIR="${1:-${WORKSPACE}/datasets/NP_benchmark_package}"
OUT_DIR="${2:-${WORKSPACE}/runs/benchmarks/np_singlepoint}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIGS="${CONFIGS:-${PACKAGE_DIR}/np_neb_benchmark_all.extxyz}"
DEVICE="${DEVICE:-cuda}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-float64}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-10}"

MODEL_LABELS="${MODEL_LABELS:-ft_best_loss scratch_best_loss ft_with_np_old_best_loss}"
MODEL_PATHS="${MODEL_PATHS:-${WORKSPACE}/models/ft_best_loss/model.model ${WORKSPACE}/models/scratch_best_loss/model.model ${WORKSPACE}/models/ft_np_baseline/model.model}"

read -r -a LABEL_ARRAY <<< "${MODEL_LABELS}"
read -r -a MODEL_ARRAY <<< "${MODEL_PATHS}"

if [[ ${#LABEL_ARRAY[@]} -ne ${#MODEL_ARRAY[@]} ]]; then
  echo "MODEL_LABELS and MODEL_PATHS must contain the same number of items." >&2
  echo "MODEL_LABELS=${MODEL_LABELS}" >&2
  echo "MODEL_PATHS=${MODEL_PATHS}" >&2
  exit 2
fi

if [[ ! -f "${CONFIGS}" ]]; then
  echo "NP benchmark extxyz not found: ${CONFIGS}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/np_neb_singlepoint_benchmark_manifest.csv"
echo "label,model,predicted_extxyz,split_metrics_csv,np_group_csv,np_summary_md" > "${MANIFEST}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

echo "Package dir: ${PACKAGE_DIR}"
echo "Configs: ${CONFIGS}"
echo "Output dir: ${OUT_DIR}"
echo "Device: ${DEVICE}"
echo "Default dtype: ${DEFAULT_DTYPE}"
echo "Models: ${MODEL_LABELS}"

for idx in "${!LABEL_ARRAY[@]}"; do
  label="${LABEL_ARRAY[$idx]}"
  model="${MODEL_ARRAY[$idx]}"
  model_dir="${OUT_DIR}/${label}"
  pred="${model_dir}/${label}_np_neb_pred.extxyz"
  split_csv="${model_dir}/${label}_np_split_metrics.csv"
  split_md="${model_dir}/${label}_np_split_metrics.md"
  group_csv="${model_dir}/${label}_np_neb_pred_neb_group_barriers.csv"
  summary_md="${model_dir}/${label}_np_neb_pred_np_neb_summary.md"

  if [[ ! -f "${model}" ]]; then
    echo "Model not found for ${label}: ${model}" >&2
    exit 2
  fi

  mkdir -p "${model_dir}"
  echo
  echo "=== ${label}: single-point prediction ==="
  "${PYTHON_BIN}" "${REPO_ROOT}/ptni_mace_workflow/evaluation/evaluate_mace_extxyz_one_by_one.py" \
    --configs "${CONFIGS}" \
    --model "${model}" \
    --output "${pred}" \
    --device "${DEVICE}" \
    --default-dtype "${DEFAULT_DTYPE}" \
    --empty-cache-every "${EMPTY_CACHE_EVERY}" \
    "${LIMIT_ARGS[@]}"

  echo
  echo "=== ${label}: structure-level scoring ==="
  "${PYTHON_BIN}" "${REPO_ROOT}/ptni_mace_workflow/evaluation/score_mace_predictions_extxyz.py" \
    --pred "${label}=${pred}" \
    --out-csv "${split_csv}" \
    --out-md "${split_md}"

  echo
  echo "=== ${label}: NP NEB triplet scoring ==="
  "${PYTHON_BIN}" "${SCRIPT_DIR}/score_np_neb_singlepoint.py" \
    --pred "${pred}" \
    --out-dir "${model_dir}" \
    --name "${label}_np_neb_pred"

  echo "${label},${model},${pred},${split_csv},${group_csv},${summary_md}" >> "${MANIFEST}"
done

echo
echo "Done."
echo "Manifest: ${MANIFEST}"
