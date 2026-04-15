#!/usr/bin/env bash

set -euo pipefail
set -o errtrace

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

trap 'ci_dump_meson_log' ERR

ci_enter_repo_root

: "${VENV_DIR:=.venv-internal-ci/full}"
: "${FULL_PYTEST_ARGS:=numpy --durations=10 --timeout=600}"

coverage_html_dir="build/coverage"
coverage_xml="build/coverage/coverage.xml"

ci_activate_python

ci_install_python_requirements requirements/build_requirements.txt
ci_install_python_requirements requirements/test_requirements.txt
if ! ci_should_install_python_deps; then
    ci_require_command spin ninja pytest
    ci_require_python_modules Cython mesonbuild mesonpy build hypothesis pytest_timeout xdist coverage pytz
fi

ci_log "Installing OpenBLAS-related system dependencies."
ci_install_system_packages \
    "gfortran libgfortran5" \
    "gcc-gfortran libgfortran" \
    "gcc-gfortran libgfortran"

ci_log "Preparing scipy-openblas pkg-config metadata."
ci_prepare_openblas32

ci_log "Installing NumPy in editable mode."
pip install -e . --no-build-isolation

ci_log "Running full unit test suite with coverage."
case " ${FULL_PYTEST_ARGS} " in
    *" --cov "*|*" --cov="*|*" --cov-report "*|*" --cov-report="*)
        printf 'FULL_PYTEST_ARGS must not include --cov/--cov-report; full.sh manages coverage itself.\n' >&2
        exit 2
        ;;
esac

read -r -a full_pytest_args <<< "${FULL_PYTEST_ARGS}"

read -r python_major python_minor <<< "$(python - <<'PY'
import sys
print(sys.version_info[0], sys.version_info[1])
PY
)"

if (( python_major > 3 || (python_major == 3 && python_minor >= 12) )); then
    case " ${FULL_PYTEST_ARGS} " in
        *" --ignore=numpy/distutils/tests "*)
            ;;
        *)
            ci_log "Python ${python_major}.${python_minor} detected; excluding numpy/distutils/tests."
            full_pytest_args+=(--ignore=numpy/distutils/tests)
            ;;
    esac
fi

python -m coverage erase
PYTHONOPTIMIZE=2 python -m coverage run --source=numpy -m pytest "${full_pytest_args[@]}"

ci_log "Generating HTML and XML coverage artifacts."
mkdir -p "${coverage_html_dir}"
python -m coverage html -d "${coverage_html_dir}"
python -m coverage xml -o "${coverage_xml}"
