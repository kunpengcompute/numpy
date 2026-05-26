#!/usr/bin/env bash

set -euo pipefail
set -o errtrace

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

trap 'ci_dump_meson_log' ERR

ci_enter_repo_root

: "${VENV_DIR:=.venv-internal-ci/full_ut}"
: "${FULL_UT_ARGS:=numpy --durations=10 --timeout=600}"
: "${FULL_UT_DESELECTS:=
numpy/_core/tests/test_multiarray.py::TestResize::test_obj_obj_shrinking[O]
numpy/_core/tests/test_multiarray.py::TestResize::test_obj_obj_shrinking[O,O]
numpy/_core/tests/test_multiarray.py::TestSizeOf::test_resize_structured[u4,f4]
numpy/_core/tests/test_multiarray.py::TestSizeOf::test_resize_structured[u4,O]
numpy/_core/tests/test_stringdtype.py::test_resize_method
numpy/linalg/tests/test_linalg.py::TestDet::test_types[complex64]
numpy/linalg/tests/test_linalg.py::TestDet::test_types[complex128]
}"
: "${FULL_UT_SKIP_EXPR:=test_obj_obj_shrinking or test_resize_structured or test_resize_method or (TestDet and test_types and (complex64 or complex128))}"
: "${FULL_UT_BUILD_ARGS:=}"

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

ci_log "Building NumPy for full UT without coverage."
read -r -a full_ut_build_args <<< "${FULL_UT_BUILD_ARGS}"
spin build --clean -- "${full_ut_build_args[@]}"

ci_log "Running full UT suite without coverage."
export PATH="${PWD}/build-install/usr/bin:${PATH}"
case " ${FULL_UT_ARGS} " in
    *" --cov "*|*" --cov="*|*" --cov-report "*|*" --cov-report="*|*" --gcov "*|*" --gcov-format "*)
        printf 'FULL_UT_ARGS must not include coverage options; full_ut.sh runs UT only.\n' >&2
        exit 2
        ;;
esac

read -r -a full_ut_args <<< "${FULL_UT_ARGS}"

if [[ -n "${FULL_UT_SKIP_EXPR}" ]]; then
    case " ${FULL_UT_ARGS} " in
        *" -k "*|*" -k="*)
            ci_log "FULL_UT_ARGS already contains -k; not adding FULL_UT_SKIP_EXPR."
            ;;
        *)
            full_ut_args+=(-k "not (${FULL_UT_SKIP_EXPR})")
            ;;
    esac
fi

while IFS= read -r deselect_nodeid; do
    [[ -n "${deselect_nodeid}" ]] || continue
    full_ut_args+=(--deselect="${deselect_nodeid}")
done <<< "${FULL_UT_DESELECTS}"
ci_apply_numpy_distutils_policy full_ut_args full_ut

spin test -- "${full_ut_args[@]}"
