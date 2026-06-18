#!/usr/bin/env bash
set -euo pipefail

# Batch run NP endpoint relaxation + MACE CI-NEB for multiple models.
# Defaults compare no-NP fine-tuned, scratch, and old with-NP fine-tuned models.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
cd "${REPO_ROOT}"
mkdir -p "${WORKSPACE_ARG}"
WORKSPACE="$(cd "${WORKSPACE_ARG}" && pwd)"

CONFIGS="${1:-${WORKSPACE}/datasets/NP_benchmark_package/np_neb_benchmark_all.extxyz}"
OUT_ROOT="${2:-${WORKSPACE}/runs/benchmarks/np_relax_neb}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-float64}"
PBC="${PBC:-false}"
WRAP_SCALED="${WRAP_SCALED:-1}"
REORDER_PATH_ATOMS="${REORDER_PATH_ATOMS:-1}"
REORDER_METRIC="${REORDER_METRIC:-no_pbc}"

MODEL_LABELS="${MODEL_LABELS:-ft_best_loss scratch_best_loss ft_with_np_old_best_loss}"
MODEL_PATHS="${MODEL_PATHS:-${WORKSPACE}/models/ft_best_loss/model.model ${WORKSPACE}/models/scratch_best_loss/model.model ${WORKSPACE}/models/ft_np_baseline/model.model}"

N_IMAGES="${N_IMAGES:-5}"
ENDPOINT_FMAX="${ENDPOINT_FMAX:-0.02}"
ENDPOINT_STEPS="${ENDPOINT_STEPS:-300}"
ENDPOINT_MAXSTEP="${ENDPOINT_MAXSTEP:-0.05}"
NEB_FMAX="${NEB_FMAX:-0.05}"
NEB_STEPS="${NEB_STEPS:-400}"
NEB_MAXSTEP="${NEB_MAXSTEP:-0.05}"
SEED_MIDDLE="${SEED_MIDDLE:-dft_ts}"

WRITE_IMAGES="${WRITE_IMAGES:-1}"
WRITE_TRAJECTORIES="${WRITE_TRAJECTORIES:-0}"
FIRE_DOWNHILL_CHECK="${FIRE_DOWNHILL_CHECK:-1}"
CLIMB="${CLIMB:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

FORCE_WARN="${FORCE_WARN:-1.0}"
DETACHED_NN="${DETACHED_NN:-4.0}"
COLLISION_NN="${COLLISION_NN:-1.8}"
DISPLACEMENT_WARN="${DISPLACEMENT_WARN:-5.0}"
NEB_SP_GAP_WARN="${NEB_SP_GAP_WARN:-0.5}"

read -r -a LABEL_ARRAY <<< "${MODEL_LABELS}"
read -r -a MODEL_ARRAY <<< "${MODEL_PATHS}"

if [[ ${#LABEL_ARRAY[@]} -ne ${#MODEL_ARRAY[@]} ]]; then
  echo "MODEL_LABELS and MODEL_PATHS must contain the same number of items." >&2
  echo "MODEL_LABELS=${MODEL_LABELS}" >&2
  echo "MODEL_PATHS=${MODEL_PATHS}" >&2
  exit 2
fi

if [[ ! -f "${CONFIGS}" ]]; then
  echo "Configs file not found: ${CONFIGS}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
MANIFEST="${OUT_ROOT}/np_neb_relax_neb_two_models_manifest.csv"
echo "label,model,out_dir,summary_csv,audit_csv,audit_md" > "${MANIFEST}"

COMMON_ARGS=(
  --configs "${CONFIGS}"
  --device "${DEVICE}"
  --default-dtype "${DEFAULT_DTYPE}"
  --pbc "${PBC}"
  --reorder-metric "${REORDER_METRIC}"
  --n-images "${N_IMAGES}"
  --endpoint-fmax "${ENDPOINT_FMAX}"
  --endpoint-steps "${ENDPOINT_STEPS}"
  --endpoint-maxstep "${ENDPOINT_MAXSTEP}"
  --neb-fmax "${NEB_FMAX}"
  --neb-steps "${NEB_STEPS}"
  --neb-maxstep "${NEB_MAXSTEP}"
  --seed-middle "${SEED_MIDDLE}"
)

if [[ "${WRITE_IMAGES}" == "1" ]]; then
  COMMON_ARGS+=(--write-images)
fi
if [[ "${WRAP_SCALED}" != "1" ]]; then
  COMMON_ARGS+=(--no-wrap-scaled)
fi
if [[ "${REORDER_PATH_ATOMS}" != "1" ]]; then
  COMMON_ARGS+=(--no-reorder-path-atoms)
fi
if [[ "${WRITE_TRAJECTORIES}" == "1" ]]; then
  COMMON_ARGS+=(--write-trajectories)
fi
if [[ "${FIRE_DOWNHILL_CHECK}" == "1" ]]; then
  COMMON_ARGS+=(--fire-downhill-check)
fi
if [[ "${CLIMB}" != "1" ]]; then
  COMMON_ARGS+=(--no-climb)
fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  COMMON_ARGS+=(--continue-on-error)
fi
if [[ -n "${MAX_GROUPS:-}" ]]; then
  COMMON_ARGS+=(--max-groups "${MAX_GROUPS}")
fi
if [[ -n "${GROUP_FILTER:-}" ]]; then
  COMMON_ARGS+=(--group-filter "${GROUP_FILTER}")
fi

echo "Configs: ${CONFIGS}"
echo "Output root: ${OUT_ROOT}"
echo "Device: ${DEVICE}"
echo "Geometry preprocessing: wrap_scaled=${WRAP_SCALED}, pbc=${PBC}, reorder_path_atoms=${REORDER_PATH_ATOMS}, reorder_metric=${REORDER_METRIC}"
echo "Models: ${MODEL_LABELS}"
echo "NEB: images=${N_IMAGES}, steps=${NEB_STEPS}, fmax=${NEB_FMAX}, maxstep=${NEB_MAXSTEP}, climb=${CLIMB}"
echo "Endpoint: steps=${ENDPOINT_STEPS}, fmax=${ENDPOINT_FMAX}, maxstep=${ENDPOINT_MAXSTEP}"
if [[ -n "${GROUP_FILTER:-}" ]]; then
  echo "Group filter: ${GROUP_FILTER}"
fi
if [[ -n "${MAX_GROUPS:-}" ]]; then
  echo "Max groups: ${MAX_GROUPS}"
fi

for idx in "${!LABEL_ARRAY[@]}"; do
  label="${LABEL_ARRAY[$idx]}"
  model="${MODEL_ARRAY[$idx]}"
  out_dir="${OUT_ROOT}/${label}"
  summary_csv="${out_dir}/np_relax_neb_summary.csv"
  audit_csv="${out_dir}/np_relax_neb_audit.csv"
  audit_md="${out_dir}/np_relax_neb_audit.md"

  if [[ ! -f "${model}" ]]; then
    echo "Model not found for ${label}: ${model}" >&2
    exit 2
  fi

  mkdir -p "${out_dir}"
  echo
  echo "=== ${label}: relax endpoints + CI-NEB ==="
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_np_neb_relax_neb.py" \
    "${COMMON_ARGS[@]}" \
    --model "${model}" \
    --out-dir "${out_dir}"

  echo
  echo "=== ${label}: audit relax+NEB output ==="
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_np_neb_relax_results.py" \
    --summary-csv "${summary_csv}" \
    --out-csv "${audit_csv}" \
    --out-md "${audit_md}" \
    --force-warn "${FORCE_WARN}" \
    --detached-nn "${DETACHED_NN}" \
    --collision-nn "${COLLISION_NN}" \
    --displacement-warn "${DISPLACEMENT_WARN}" \
    --neb-sp-gap-warn "${NEB_SP_GAP_WARN}"

  echo "${label},${model},${out_dir},${summary_csv},${audit_csv},${audit_md}" >> "${MANIFEST}"
done

echo
echo "Done."
echo "Manifest: ${MANIFEST}"
