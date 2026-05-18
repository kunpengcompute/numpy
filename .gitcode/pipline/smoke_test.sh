#!/usr/bin/env bash

set -euo pipefail
set -o errtrace

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

trap 'ci_dump_meson_log' ERR

ci_enter_repo_root

: "${VENV_DIR:=.venv-internal-ci/smoke_test}"
: "${SMOKE_TEST_ARGS:=--durations=10 --timeout=600}"

ci_activate_python

ci_install_python_requirements requirements/build_requirements.txt
if ! ci_should_install_python_deps; then
    ci_require_command spin ninja pytest
    ci_require_python_modules Cython mesonbuild mesonpy build hypothesis pytest_timeout xdist coverage pytz
fi

ci_log "Building NumPy for smoke test."
spin build --clean -- -Dallow-noblas=true -Dcpu-baseline=none -Dcpu-dispatch=none

ci_install_python_requirements requirements/test_requirements.txt

ci_log "Running smoke test suite."
read -r -a smoke_test_args <<< "${SMOKE_TEST_ARGS}"

read -r python_major python_minor <<< "$(python - <<'PY'
import sys
print(sys.version_info[0], sys.version_info[1])
PY
)"

if (( python_major > 3 || (python_major == 3 && python_minor >= 12) )); then
    case " ${SMOKE_TEST_ARGS} " in
        *" --ignore=numpy/distutils/tests "*)
            ;;
        *)
            ci_log "Python ${python_major}.${python_minor} detected; excluding numpy/distutils/tests."
            smoke_test_args+=(--ignore=numpy/distutils/tests)
            ;;
    esac
else
    export SETUPTOOLS_USE_DISTUTILS="${SETUPTOOLS_USE_DISTUTILS:-stdlib}"
    ci_log "Using SETUPTOOLS_USE_DISTUTILS=${SETUPTOOLS_USE_DISTUTILS} for numpy.distutils tests."
    smoke_test_args+=(-W "ignore:Reliance on distutils from stdlib is deprecated:UserWarning")
fi

spin test -- "${smoke_test_args[@]}"
