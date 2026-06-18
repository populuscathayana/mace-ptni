#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash ptni_mace_workflow/benchmarks/run_benchmark_suite.sh \
    --workspace mace_workspace \
    --model-tag ft_best_loss \
    --suite lattice,strained_neb,pt111_pes,np_singlepoint,np_relax_neb

Options:
  --workspace DIR       Runtime workspace. Default: MACE_WORKSPACE or mace_workspace.
  --model-tag TAG       Model tag under workspace/models/<TAG>/model.model.
  --model PATH          Explicit .model path. Overrides --model-tag path lookup.
  --suite LIST          Comma-separated tasks. Default: lattice,strained_neb,pt111_pes.
  --device DEVICE       cuda or cpu. Default: cuda.
  --default-dtype TYPE  Default dtype. Default: float64.
  --smoke               Reduce selected tasks to tiny sanity checks where supported.

Suite names:
  lattice, strained_neb, pt111_pes, np_singlepoint, np_relax_neb
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ARG="${MACE_WORKSPACE:-mace_workspace}"
MODEL_TAG=""
MODEL_PATH=""
SUITE="lattice,strained_neb,pt111_pes"
DEVICE="${DEVICE:-cuda}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-float64}"
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE_ARG="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --default-dtype) DEFAULT_DTYPE="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
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
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ "${MODEL_PATH}" == *.pt ]]; then
  echo "Refusing .pt checkpoint for benchmark: ${MODEL_PATH}" >&2
  echo "Export it to .model first." >&2
  exit 2
fi

PT_SCAN="${PT_SCAN:-3.75:4.15:0.01}"
NI_SCAN="${NI_SCAN:-3.35:3.75:0.01}"
FIT_WINDOW="${FIT_WINDOW:-0.08}"
STRAIN_SCAN="${STRAIN_SCAN:--3:3:1}"
NEB_MODE="${NEB_MODE:-isotropic}"
NEB_IMAGES="${NEB_IMAGES:-5}"
ENDPOINT_FMAX="${ENDPOINT_FMAX:-0.02}"
ENDPOINT_STEPS="${ENDPOINT_STEPS:-300}"
NEB_FMAX="${NEB_FMAX:-0.05}"
NEB_STEPS="${NEB_STEPS:-400}"
PT111_PATTERN="${PT111_PATTERN:-hex_point_[0-9]*/POSCAR}"
PT111_OPTIMIZER="${PT111_OPTIMIZER:-BFGS}"
PT111_FMAX="${PT111_FMAX:-1e-4}"
PT111_STEPS="${PT111_STEPS:-1000}"
PT111_RELAX_MODE="${PT111_RELAX_MODE:-force}"

if [[ "${SMOKE}" == "1" ]]; then
  PT_SCAN="${PT_SCAN_SMOKE:-3.95:4.05:0.05}"
  NI_SCAN="${NI_SCAN_SMOKE:-3.45:3.65:0.05}"
  STRAIN_SCAN="${STRAIN_SCAN_SMOKE:-0:0:1}"
  NEB_IMAGES="${NEB_IMAGES_SMOKE:-3}"
  ENDPOINT_STEPS="${ENDPOINT_STEPS_SMOKE:-1}"
  NEB_STEPS="${NEB_STEPS_SMOKE:-1}"
  PT111_PATTERN="${PT111_PATTERN_SMOKE:-hex_point_000/POSCAR}"
  PT111_STEPS="${PT111_STEPS_SMOKE:-1}"
  export LIMIT="${LIMIT:-3}"
  export MAX_GROUPS="${MAX_GROUPS:-1}"
fi

MANIFEST="${WORKSPACE}/runs/benchmarks/benchmark_suite_manifest.csv"
mkdir -p "$(dirname "${MANIFEST}")"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "model_tag,benchmark,model,output_dir,device,default_dtype,smoke" > "${MANIFEST}"
fi

contains_suite() {
  local needle="$1"
  [[ ",${SUITE}," == *",${needle},"* ]]
}

record_manifest() {
  local benchmark="$1"
  local out_dir="$2"
  echo "${MODEL_TAG},${benchmark},${MODEL_PATH},${out_dir},${DEVICE},${DEFAULT_DTYPE},${SMOKE}" >> "${MANIFEST}"
}

echo "Benchmark suite:"
echo "  WORKSPACE=${WORKSPACE}"
echo "  MODEL_TAG=${MODEL_TAG}"
echo "  MODEL=${MODEL_PATH}"
echo "  SUITE=${SUITE}"
echo "  DEVICE=${DEVICE}"
echo "  SMOKE=${SMOKE}"

if contains_suite lattice; then
  OUT_DIR="${WORKSPACE}/runs/benchmarks/lattice/${MODEL_TAG}"
  mkdir -p "${OUT_DIR}"
  echo
  echo "=== lattice: Pt fcc ==="
  python "${SCRIPT_DIR}/lattice/pt_fcc_lattice_constant_mace.py" \
    --model "${MODEL_PATH}" \
    --scan "${PT_SCAN}" \
    --fit-window "${FIT_WINDOW}" \
    --device "${DEVICE}" \
    --default-dtype "${DEFAULT_DTYPE}" \
    --out-dir "${OUT_DIR}/pt_fcc_lattice_test" \
    --name "${MODEL_TAG}_pt_fcc" \
    --plot
  echo
  echo "=== lattice: Ni fcc ==="
  python "${SCRIPT_DIR}/lattice/ni_fcc_lattice_constant_mace.py" \
    --model "${MODEL_PATH}" \
    --scan "${NI_SCAN}" \
    --fit-window "${FIT_WINDOW}" \
    --device "${DEVICE}" \
    --default-dtype "${DEFAULT_DTYPE}" \
    --out-dir "${OUT_DIR}/ni_fcc_lattice_test" \
    --name "${MODEL_TAG}_ni_fcc" \
    --plot
  record_manifest lattice "${OUT_DIR}"
fi

if contains_suite strained_neb; then
  OUT_DIR="${WORKSPACE}/runs/benchmarks/strained_neb/${MODEL_TAG}"
  IS_POSCAR="${IS_POSCAR:-${WORKSPACE}/inputs/strained_neb/POSCAR-is}"
  TS_POSCAR="${TS_POSCAR:-${WORKSPACE}/inputs/strained_neb/POSCAR-ts}"
  FS_POSCAR="${FS_POSCAR:-${WORKSPACE}/inputs/strained_neb/POSCAR-fs}"
  echo
  echo "=== strained_neb ==="
  python "${SCRIPT_DIR}/strained_neb/strained_neb_activation_mace.py" \
    --is-poscar "${IS_POSCAR}" \
    --ts-poscar "${TS_POSCAR}" \
    --fs-poscar "${FS_POSCAR}" \
    --model "${MODEL_PATH}" \
    --strain="${STRAIN_SCAN}" \
    --mode "${NEB_MODE}" \
    --n-images "${NEB_IMAGES}" \
    --endpoint-fmax "${ENDPOINT_FMAX}" \
    --endpoint-steps "${ENDPOINT_STEPS}" \
    --neb-fmax "${NEB_FMAX}" \
    --neb-steps "${NEB_STEPS}" \
    --device "${DEVICE}" \
    --default-dtype "${DEFAULT_DTYPE}" \
    --out-dir "${OUT_DIR}" \
    --name "${MODEL_TAG}_strained_neb" \
    --write-images
  record_manifest strained_neb "${OUT_DIR}"
fi

if contains_suite pt111_pes; then
  OUT_DIR="${WORKSPACE}/runs/benchmarks/pt111_pes/${MODEL_TAG}"
  PES_DIR="${OUT_DIR}/pt111_adatom_pes_forcefine"
  PES_NAME="${MODEL_TAG}_pt111_forcefine"
  PT111_INPUT_DIR="${PT111_INPUT_DIR:-${WORKSPACE}/inputs/pt111}"
  ORIGIN_DIR="${ORIGIN_DIR:-${WORKSPACE}/inputs/pt111/hex_point_origin}"
  echo
  echo "=== pt111_pes ==="
  python "${SCRIPT_DIR}/pt111_pes/pt111_adatom_pes_mace.py" \
    --input-dir "${PT111_INPUT_DIR}" \
    --pattern "${PT111_PATTERN}" \
    --model "${MODEL_PATH}" \
    --out-dir "${PES_DIR}" \
    --name "${PES_NAME}" \
    --device "${DEVICE}" \
    --default-dtype "${DEFAULT_DTYPE}" \
    --relax-mode "${PT111_RELAX_MODE}" \
    --optimizer "${PT111_OPTIMIZER}" \
    --fmax "${PT111_FMAX}" \
    --steps "${PT111_STEPS}" \
    --plot \
    --write-relaxed \
    --overwrite
  if [[ -d "${ORIGIN_DIR}" && -f "${PES_DIR}/${PES_NAME}_results.csv" ]]; then
    python "${SCRIPT_DIR}/pt111_pes/add_pt111_origin_reference.py" \
      --csv "${PES_DIR}/${PES_NAME}_results.csv" \
      --origin-dir "${ORIGIN_DIR}" \
      --model "${MODEL_PATH}" \
      --device "${DEVICE}" \
      --default-dtype "${DEFAULT_DTYPE}" \
      --output "${PES_DIR}/${PES_NAME}_origin_ref_results.csv"
    python "${SCRIPT_DIR}/pt111_pes/replot_pt111_adatom_pes.py" \
      --csv "${PES_DIR}/${PES_NAME}_origin_ref_results.csv" \
      --out-dir "${PES_DIR}/origin_ref_replots" \
      --name "${PES_NAME}_origin_ref_zero_min" \
      --mace-key mace_origin_ref_meV \
      --dft-key dft_origin_ref_meV \
      --diff-key mace_minus_dft_origin_ref_meV \
      --energy-label "E(point)-E(origin) (meV)" \
      --title-prefix "origin-referenced PES" \
      --top-n 12 \
      --shift-shared-min-to-zero
  fi
  record_manifest pt111_pes "${OUT_DIR}"
fi

if contains_suite np_singlepoint; then
  echo
  echo "=== np_singlepoint ==="
  MODEL_LABELS="${MODEL_TAG}" MODEL_PATHS="${MODEL_PATH}" DEVICE="${DEVICE}" DEFAULT_DTYPE="${DEFAULT_DTYPE}" \
    bash "${SCRIPT_DIR}/np_neb/run_np_neb_singlepoint_benchmark.sh" \
      "${WORKSPACE}/datasets/NP_benchmark_package" \
      "${WORKSPACE}/runs/benchmarks/np_singlepoint"
  record_manifest np_singlepoint "${WORKSPACE}/runs/benchmarks/np_singlepoint/${MODEL_TAG}"
fi

if contains_suite np_relax_neb; then
  echo
  echo "=== np_relax_neb ==="
  MODEL_LABELS="${MODEL_TAG}" MODEL_PATHS="${MODEL_PATH}" DEVICE="${DEVICE}" DEFAULT_DTYPE="${DEFAULT_DTYPE}" \
    bash "${SCRIPT_DIR}/np_neb/run_np_neb_relax_neb_two_models.sh" \
      "${WORKSPACE}/datasets/NP_benchmark_package/np_neb_benchmark_all.extxyz" \
      "${WORKSPACE}/runs/benchmarks/np_relax_neb"
  record_manifest np_relax_neb "${WORKSPACE}/runs/benchmarks/np_relax_neb/${MODEL_TAG}"
fi

echo
echo "Benchmark suite complete."
echo "Manifest: ${MANIFEST}"
