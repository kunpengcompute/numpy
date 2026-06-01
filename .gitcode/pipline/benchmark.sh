#!/usr/bin/env bash

set -euo pipefail
set -o errtrace

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

trap 'ci_dump_meson_log' ERR

ci_enter_repo_root

: "${VENV_DIR:=.venv-internal-ci/benchmark}"
: "${BENCHMARK_ARGS:=--quick}"

ci_activate_python

ci_log "Installing benchmark system dependencies."
ci_install_system_packages \
    "libopenblas-dev ninja-build" \
    "openblas-devel ninja-build" \
    "openblas-devel ninja-build"

if [[ "${SKIP_SYSTEM_DEPS:-1}" == "1" ]]; then
    ci_log "Preparing scipy-openblas fallback because system dependencies are skipped."
    ci_prepare_openblas32
fi

ci_install_python_requirements requirements/build_requirements.txt
ci_install_python_packages "asv<0.6.5" virtualenv packaging
if ! ci_should_install_python_deps; then
    ci_require_command spin ninja asv
    ci_require_python_modules Cython mesonbuild mesonpy build virtualenv packaging
fi

ci_log "Building NumPy for benchmark job."
if [[ "${SKIP_SYSTEM_DEPS:-1}" == "1" ]]; then
    spin build --with-scipy-openblas=32 -- -Dcpu-dispatch=none
else
    spin build -- -Dcpu-dispatch=none
fi

ci_log "Recording ASV machine information."
ci_run_via_script_shell "cd \"$(ci_repo_root)\" && asv machine --yes --config benchmarks/asv.conf.json"

ci_log "Running quick ASV benchmark suite."
ci_run_via_script_shell "cd \"$(ci_repo_root)\" && spin bench ${BENCHMARK_ARGS}"
