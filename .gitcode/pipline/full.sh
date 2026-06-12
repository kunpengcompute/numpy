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
: "${FULL_PYTEST_DESELECTS:=
numpy/_core/tests/test_multiarray.py::TestResize::test_obj_obj_shrinking[O]
numpy/_core/tests/test_multiarray.py::TestResize::test_obj_obj_shrinking[O,O]
numpy/_core/tests/test_multiarray.py::TestSizeOf::test_resize_structured[u4,f4]
numpy/_core/tests/test_multiarray.py::TestSizeOf::test_resize_structured[u4,O]
numpy/_core/tests/test_stringdtype.py::test_resize_method
numpy/linalg/tests/test_linalg.py::TestDet::test_types[complex64]
numpy/linalg/tests/test_linalg.py::TestDet::test_types[complex128]
}"
: "${FULL_PYTEST_SKIP_EXPR:=test_obj_obj_shrinking or test_resize_structured or test_resize_method or (TestDet and test_types and (complex64 or complex128))}"
: "${FULL_PYTHONOPTIMIZE:=}"

coverage_html_dir="build/coverage"
coverage_xml="build/coverage/coverage.xml"
coverage_data="$(pwd)/${coverage_html_dir}/.coverage"
python_coverage_omit="*/tests/*,*/conftest.py"
build_dir="${SPIN_BUILD_DIR:-build}"
gcov_format="${FULL_GCOV_FORMAT:-xml}"
c_coverage_xml="${build_dir}/meson-logs/coverage.xml"
: "${PYTHON_LINE_COVERAGE_MIN:=70}"
: "${PYTHON_BRANCH_COVERAGE_MIN:=40}"
: "${C_LINE_COVERAGE_MIN:=70}"
: "${C_BRANCH_COVERAGE_MIN:=40}"
: "${FULL_GCC_COVERAGE_ICE_WORKAROUND:=1}"

ci_compiler_accepts_flag() {
    local compiler="$1"
    local language="$2"
    local flag="$3"
    local -a compiler_args

    read -r -a compiler_args <<< "${compiler}"
    printf 'int main(void) { return 0; }\n' \
        | "${compiler_args[@]}" -x "${language}" "${flag}" -c -o /dev/null - >/dev/null 2>&1
}

ci_append_supported_coverage_warning_flags() {
    local warning_flag="-Wno-error=coverage-invalid-line-number"
    local cc_bin="${CC:-cc}"
    local cxx_bin="${CXX:-c++}"

    if ci_compiler_accepts_flag "${cc_bin}" c "${warning_flag}"; then
        export CFLAGS="${CFLAGS:-} ${warning_flag}"
        ci_log "Added supported C coverage warning flag: ${warning_flag}"
    else
        ci_log "Skipping unsupported C coverage warning flag for ${cc_bin}: ${warning_flag}"
    fi

    if ci_compiler_accepts_flag "${cxx_bin}" c++ "${warning_flag}"; then
        export CXXFLAGS="${CXXFLAGS:-} ${warning_flag}"
        ci_log "Added supported C++ coverage warning flag: ${warning_flag}"
    else
        ci_log "Skipping unsupported C++ coverage warning flag for ${cxx_bin}: ${warning_flag}"
    fi
}

ci_append_coverage_link_flags() {
    local link_flag="--coverage"

    export LDFLAGS="${LDFLAGS:-} ${link_flag}"
    ci_log "Added coverage link flag for subprocess extension builds: ${link_flag}"
}

ci_configure_gcov_tool() {
    local cc_path
    local gcov_path
    local cc_major
    local candidate
    local -a compiler_args
    local -a gcov_candidates=()

    [[ -z "${GCOV:-}" ]] || return 0

    read -r -a compiler_args <<< "${CC:-cc}"
    cc_path="$(command -v "${compiler_args[0]}" || true)"
    if [[ -n "${cc_path}" ]]; then
        gcov_candidates+=("$(dirname "${cc_path}")/gcov")
    fi

    cc_major="$(ci_compiler_gcc_major "${CC:-cc}" || true)"
    if [[ -n "${cc_major}" ]]; then
        gcov_candidates+=("/opt/rh/gcc-toolset-${cc_major}/root/usr/bin/gcov")
    fi

    for candidate in "${gcov_candidates[@]}"; do
        [[ -x "${candidate}" ]] || continue
        if [[ -n "${cc_major}" ]] && ! "${candidate}" --version 2>/dev/null | head -n 1 | grep -q " ${cc_major}\\."; then
            continue
        fi

        export GCOV="${candidate}"
        ci_log "Using gcov from compiler toolchain: ${GCOV}"
        return 0
    done

    ci_log "Using default gcov from PATH."
}

ci_compiler_gcc_major() {
    local compiler="$1"
    local macro
    local -a compiler_args

    read -r -a compiler_args <<< "${compiler}"

    macro="$(printf '\n' | "${compiler_args[@]}" -dM -E - 2>/dev/null \
        | awk '
            $2 == "__clang__" { clang = 1 }
            $2 == "__GNUC__" { gcc_major = $3 }
            END {
                if (!clang && gcc_major ~ /^[0-9]+$/) {
                    print gcc_major
                } else {
                    exit 1
                }
            }
        ')" || return 1
    printf '%s\n' "${macro}"
}

ci_gcc_coverage_ice_workaround_needed() {
    local cc_major

    [[ "${FULL_GCC_COVERAGE_ICE_WORKAROUND}" == "1" ]] || return 1

    cc_major="$(ci_compiler_gcc_major "${CC:-cc}" || true)"
    [[ -n "${cc_major}" ]] && (( cc_major < 12 ))
}

ci_configure_gcc_coverage_ice_workaround() {
    local meson_py="vendored-meson/meson/meson.py"

    ci_gcc_coverage_ice_workaround_needed || return 0

    ci_log "Configuring Meson with -Doptimization=0 to avoid old GCC coverage ICE."
    if [[ -f "${build_dir}/meson-info/intro-buildoptions.json" ]]; then
        python "${meson_py}" configure "${build_dir}" -Doptimization=0
    else
        python "${meson_py}" setup "${build_dir}" --prefix=/usr -Db_coverage=true -Doptimization=0
    fi
}

ci_activate_python

ci_install_python_requirements requirements/build_requirements.txt
ci_install_python_requirements requirements/test_requirements.txt
ci_install_python_packages gcovr
if ! ci_should_install_python_deps; then
    ci_require_command spin ninja pytest gcov gcovr
    ci_require_python_modules Cython mesonbuild mesonpy build hypothesis pytest_timeout xdist coverage pytest_cov pytz gcovr
fi

ci_log "Installing OpenBLAS-related system dependencies."
ci_install_system_packages \
    "gfortran libgfortran5" \
    "gcc-gfortran libgfortran" \
    "gcc-gfortran libgfortran"

ci_log "Preparing scipy-openblas pkg-config metadata."
ci_prepare_openblas32

ci_log "Patching scipy-openblas.pc to include Libs for shared linking."
_pc_file=".openblas/scipy-openblas.pc"
if [[ -f "${_pc_file}" ]]; then
    if grep -q '^Libs:$' "${_pc_file}" || grep -q '^Libs: *$' "${_pc_file}"; then
        _libdir="$(grep '^libdir=' "${_pc_file}" | cut -d= -f2-)"
        sed -i "s|^Libs: *$|Libs: -L${_libdir} -lscipy_openblas|" "${_pc_file}"
        ci_log "Fixed empty Libs field in ${_pc_file}."
    fi
    _libdir="$(grep '^libdir=' "${_pc_file}" | cut -d= -f2-)"
    if [[ -n "${_libdir}" && -d "${_libdir}" ]]; then
        export LD_LIBRARY_PATH="${_libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        ci_log "Added ${_libdir} to LD_LIBRARY_PATH."
    fi
fi

ci_log "Running full unit test suite with Python and C/C++ coverage."
case " ${FULL_PYTEST_ARGS} " in
    *" --cov "*|*" --cov="*|*" --cov-report "*|*" --cov-report="*)
        printf 'FULL_PYTEST_ARGS must not include --cov/--cov-report; full.sh manages coverage itself.\n' >&2
        exit 2
        ;;
esac

case "${gcov_format}" in
    html|xml|text|sonarqube)
        ;;
    *)
        printf 'Unsupported FULL_GCOV_FORMAT: %s (expected html, xml, text, or sonarqube)\n' "${gcov_format}" >&2
        exit 2
        ;;
esac

read -r -a full_pytest_args <<< "${FULL_PYTEST_ARGS}"

if [[ -n "${FULL_PYTEST_SKIP_EXPR}" ]]; then
    case " ${FULL_PYTEST_ARGS} " in
        *" -k "*|*" -k="*)
            ci_log "FULL_PYTEST_ARGS already contains -k; not adding FULL_PYTEST_SKIP_EXPR."
            ;;
        *)
            full_pytest_args+=(-k "not (${FULL_PYTEST_SKIP_EXPR})")
            ;;
    esac
fi

while IFS= read -r deselect_nodeid; do
    [[ -n "${deselect_nodeid}" ]] || continue
    full_pytest_args+=(--deselect="${deselect_nodeid}")
done <<< "${FULL_PYTEST_DESELECTS}"
ci_apply_numpy_distutils_policy full_pytest_args full

ci_log "Preparing coverage artifact directories."
rm -rf -- "${coverage_html_dir}" "${coverage_xml}"
mkdir -p "${coverage_html_dir}"
COVERAGE_FILE="${coverage_data}" python -m coverage erase

ci_log "Writing gcovr configuration."
cat > gcovr.cfg <<'EOF'
# Ignore third-party code under subprojects to reduce noise in C/C++ coverage.
exclude = subprojects/.*

# Ignore generated build outputs and vendored SIMD/sort implementation details
# that are architecture-dependent and make cross-arch coverage thresholds noisy.
exclude = build/.*
exclude = numpy/_core/src/highway/.*
exclude = numpy/_core/src/npysort/.*

# Only ignore negative hit-count anomalies from optimized code; surface all
# other gcov parse failures so incomplete XML does not silently pass thresholds.
gcov-ignore-parse-errors = negative_hits.warn_once_per_file
EOF

ci_append_supported_coverage_warning_flags
ci_append_coverage_link_flags
ci_configure_gcov_tool
if [[ -n "${GCOV:-}" ]]; then
    printf 'gcov-executable = %s\n' "${GCOV}" >> gcovr.cfg
fi

ci_log "Running spin test with Python and C/C++ coverage."
find "${build_dir}" -name '*.gcda' -delete 2>/dev/null || true
ci_configure_gcc_coverage_ice_workaround
if [[ -n "${FULL_PYTHONOPTIMIZE}" ]]; then
    export PYTHONOPTIMIZE="${FULL_PYTHONOPTIMIZE}"
    ci_log "Running with PYTHONOPTIMIZE=${PYTHONOPTIMIZE}."
else
    unset PYTHONOPTIMIZE
    ci_log "Running without PYTHONOPTIMIZE for coverage collection."
fi
COVERAGE_FILE="${coverage_data}" spin test --gcov --gcov-format "${gcov_format}" -m full -- \
    "${full_pytest_args[@]}" \
    --cov=numpy \
    --cov-branch \
    --cov-report=

ci_log "Generating Python coverage terminal report."
COVERAGE_FILE="${coverage_data}" python -m coverage report --omit="${python_coverage_omit}"

ci_log "Generating Python HTML coverage artifact."
COVERAGE_FILE="${coverage_data}" python -m coverage html --omit="${python_coverage_omit}" -d "${coverage_html_dir}"

ci_log "Generating Python XML coverage artifact."
COVERAGE_FILE="${coverage_data}" python -m coverage xml --omit="${python_coverage_omit}" -o "${coverage_xml}"

ci_log "Regenerating C/C++ XML coverage with multi-target merge."
rm -f "${c_coverage_xml}"
gcovr_stderr_file="$(mktemp)"
python "${SCRIPT_DIR}/../../merge_coverage.py" "${build_dir}" "${c_coverage_xml}" 2>"${gcovr_stderr_file}"
if [[ -s "${gcovr_stderr_file}" ]]; then
    ci_log "WARNING: gcovr reported parse errors (see below). Coverage XML may be incomplete."
    cat "${gcovr_stderr_file}" >&2
fi
rm -f "${gcovr_stderr_file}"

ci_log "Checking Python and C/C++ coverage thresholds."
python - "${coverage_xml}" "${c_coverage_xml}" \
    "${PYTHON_LINE_COVERAGE_MIN}" "${PYTHON_BRANCH_COVERAGE_MIN}" \
    "${C_LINE_COVERAGE_MIN}" "${C_BRANCH_COVERAGE_MIN}" <<'PY'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def read_rate(xml_path, attr):
    path = Path(xml_path)
    if not path.is_file():
        raise SystemExit(f"Coverage XML not found: {path}")

    root = ET.parse(path).getroot()
    value = root.attrib.get(attr)
    if value is None:
        raise SystemExit(f"Coverage XML missing {attr}: {path}")
    return float(value) * 100.0


def check(label, xml_path, line_min, branch_min):
    line = read_rate(xml_path, "line-rate")
    branch = read_rate(xml_path, "branch-rate")
    line_min = float(line_min)
    branch_min = float(branch_min)

    print(f"{label} coverage:")
    print(f"  line: {line:.2f}% >= {line_min:.2f}%")
    print(f"  branch: {branch:.2f}% >= {branch_min:.2f}%")

    failures = []
    if line < line_min:
        failures.append(
            f"{label} line coverage {line:.2f}% is below required {line_min:.2f}%"
        )
    if branch < branch_min:
        failures.append(
            f"{label} branch coverage {branch:.2f}% is below required {branch_min:.2f}%"
        )
    return failures


python_xml, c_xml, py_line_min, py_branch_min, c_line_min, c_branch_min = sys.argv[1:]
failures = []
failures.extend(check("Python", python_xml, py_line_min, py_branch_min))
failures.extend(check("C/C++", c_xml, c_line_min, c_branch_min))

if failures:
    print("\nCoverage threshold check failed:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    raise SystemExit(1)
PY
