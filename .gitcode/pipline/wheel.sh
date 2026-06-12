#!/usr/bin/env bash

set -euo pipefail
set -o errtrace

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

trap 'ci_dump_meson_log' ERR

ci_enter_repo_root

: "${WHEEL_OUTPUT_DIR:=wheelhouse}"
: "${WHEEL_BUILD_DIR:=build/wheel-raw}"

if [[ "${USE_VENV:-0}" == "1" ]]; then
    printf 'wheel.sh requires an activated conda environment; USE_VENV=1 is not supported.\n' >&2
    exit 2
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    printf 'wheel.sh requires an activated conda environment (CONDA_PREFIX is not set).\n' >&2
    exit 2
fi

ci_activate_python
ci_require_command pkg-config patchelf

ci_install_python_requirements requirements/build_requirements.txt
ci_install_python_packages auditwheel
ci_require_command auditwheel
if ! ci_should_install_python_deps; then
    ci_require_python_modules Cython mesonbuild mesonpy build
fi

python - "${CONDA_PREFIX}" <<'PY'
import pathlib
import sys

python_prefix = pathlib.Path(sys.prefix).resolve()
conda_prefix = pathlib.Path(sys.argv[1]).resolve()
if python_prefix != conda_prefix:
    raise SystemExit(
        f"Active python prefix {python_prefix} does not match CONDA_PREFIX {conda_prefix}"
    )
PY

conda_pkgconfig="${CONDA_PREFIX}/lib/pkgconfig"
if [[ ! -d "${conda_pkgconfig}" ]]; then
    printf 'Conda pkg-config directory not found: %s\n' "${conda_pkgconfig}" >&2
    exit 1
fi

export PKG_CONFIG_PATH="${conda_pkgconfig}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if ! pkg-config --exists openblas; then
    printf 'OpenBLAS was not found in the active conda environment via pkg-config.\n' >&2
    printf 'Expected an LP64 openblas package under CONDA_PREFIX=%s.\n' "${CONDA_PREFIX}" >&2
    exit 1
fi

openblas_pcdir="$(pkg-config --variable=pcfiledir openblas)"
case "${openblas_pcdir}/" in
    "${CONDA_PREFIX}/"*)
        ;;
    *)
        printf 'OpenBLAS pkg-config metadata did not come from the active conda environment: %s\n' \
            "${openblas_pcdir}" >&2
        exit 1
        ;;
esac

ci_log "Using conda LP64 OpenBLAS: $(pkg-config --modversion openblas)"
ci_log "Building wheel with Python: $(python -c 'import platform, sys; print(sys.version.split()[0], platform.machine())')"

rm -rf -- "${WHEEL_BUILD_DIR}"
mkdir -p "${WHEEL_BUILD_DIR}" "${WHEEL_OUTPUT_DIR}"
find "${WHEEL_OUTPUT_DIR}" -maxdepth 1 -type f -name 'numpy-*.whl' -delete

ci_log "Building raw wheel into ${WHEEL_BUILD_DIR}."
python -m build --wheel --no-isolation -o "${WHEEL_BUILD_DIR}" \
    -Csetup-args=-Dblas=openblas \
    -Csetup-args=-Dlapack=openblas \
    -Csetup-args=-Dallow-noblas=false

shopt -s nullglob
raw_wheels=("${WHEEL_BUILD_DIR}"/numpy-*.whl)
shopt -u nullglob
if [[ "${#raw_wheels[@]}" -ne 1 ]]; then
    printf 'Expected exactly one raw NumPy wheel in %s, found %s.\n' \
        "${WHEEL_BUILD_DIR}" "${#raw_wheels[@]}" >&2
    exit 1
fi

ci_log "Repairing wheel and bundling required shared libraries into ${WHEEL_OUTPUT_DIR}."
auditwheel repair "${raw_wheels[0]}" --wheel-dir "${WHEEL_OUTPUT_DIR}"

shopt -s nullglob
repaired_wheels=("${WHEEL_OUTPUT_DIR}"/numpy-*.whl)
shopt -u nullglob
if [[ "${#repaired_wheels[@]}" -ne 1 ]]; then
    printf 'Expected exactly one repaired NumPy wheel in %s, found %s.\n' \
        "${WHEEL_OUTPUT_DIR}" "${#repaired_wheels[@]}" >&2
    exit 1
fi

ci_log "Final wheel artifact: ${repaired_wheels[0]}"
