#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

ci_enter_repo_root

: "${VENV_DIR:=.venv-internal-ci/incremental_coverage}"
: "${COMPARE_BRANCH:=origin/main}"
: "${DIFF_COVER_FAIL_UNDER:=0}"
: "${DIFF_COVER_SHOW_FILES:=0}"
: "${INCREMENTAL_COVERAGE_REPORT_DIR:=build/incremental_coverage}"

python_coverage_xml="build/coverage/coverage.xml"
c_coverage_xml="build/meson-logs/coverage.xml"

ci_activate_python

missing_coverage=()
[[ -f "${python_coverage_xml}" ]] || missing_coverage+=("${python_coverage_xml}")
[[ -f "${c_coverage_xml}" ]] || missing_coverage+=("${c_coverage_xml}")

if ((${#missing_coverage[@]} > 0)); then
    if [[ "${RUN_FULL_JOB_IF_MISSING:-0}" == "1" ]]; then
        ci_log "Coverage XML missing, running full coverage job first: ${missing_coverage[*]}"
        bash "${SCRIPT_DIR}/full.sh"
    else
        printf 'Coverage XML not found:\n' >&2
        printf '  %s\n' "${missing_coverage[@]}" >&2
        printf 'Run .gitcode/pipline/full.sh first or set RUN_FULL_JOB_IF_MISSING=1.\n' >&2
        exit 1
    fi
fi

missing_coverage=()
[[ -f "${python_coverage_xml}" ]] || missing_coverage+=("${python_coverage_xml}")
[[ -f "${c_coverage_xml}" ]] || missing_coverage+=("${c_coverage_xml}")
if ((${#missing_coverage[@]} > 0)); then
    printf 'Coverage XML still not found after full coverage job:\n' >&2
    printf '  %s\n' "${missing_coverage[@]}" >&2
    exit 1
fi

ci_install_python_packages diff-cover
if ! ci_should_install_python_deps; then
    ci_require_command diff-cover
fi

ci_log "Remapping build-install paths in Python coverage XML to source paths."
python - "${python_coverage_xml}" <<'REMAP_PY'
import re
import sys
from pathlib import Path

xml_path = Path(sys.argv[1])
content = xml_path.read_text(encoding="utf-8")

prefix_match = re.search(
    r'filename="(build-install/.+?/site-packages/)', content
)
if not prefix_match:
    print("No build-install prefix in coverage XML, skipping remap.", file=sys.stderr)
    sys.exit(0)

build_prefix = prefix_match.group(1)

content = content.replace(f'filename="{build_prefix}', 'filename="')

xml_path.write_text(content, encoding="utf-8")
print(f"Remapped {build_prefix!r} -> '' in {xml_path}")
REMAP_PY

if git rev-parse --verify "${COMPARE_BRANCH}" >/dev/null 2>&1; then
    ci_log "Using compare branch ${COMPARE_BRANCH}."
else
    ci_log "Compare branch ${COMPARE_BRANCH} not found locally, trying to fetch it."
    if [[ "${COMPARE_BRANCH}" == origin/* ]]; then
        git fetch origin "${COMPARE_BRANCH#origin/}"
    else
        git fetch origin "${COMPARE_BRANCH}"
    fi
fi

case "${DIFF_COVER_SHOW_FILES}" in
    0|1)
        ;;
    *)
        printf 'Unsupported DIFF_COVER_SHOW_FILES: %s (expected 0 or 1)\n' "${DIFF_COVER_SHOW_FILES}" >&2
        exit 2
        ;;
esac

mkdir -p "${INCREMENTAL_COVERAGE_REPORT_DIR}"

run_diff_cover() {
    local label="$1"
    local xml_path="$2"
    local report_slug="$3"
    local json_report="${INCREMENTAL_COVERAGE_REPORT_DIR}/${report_slug}.json"
    local text_report="${INCREMENTAL_COVERAGE_REPORT_DIR}/${report_slug}.txt"
    local status

    ci_log "Computing ${label} incremental coverage."

    set +e
    diff-cover "${xml_path}" \
        --compare-branch "${COMPARE_BRANCH}" \
        --fail-under "${DIFF_COVER_FAIL_UNDER}" \
        --format "json:${json_report}" \
        > "${text_report}"
    status=$?
    set -e

    if [[ "${DIFF_COVER_SHOW_FILES}" == "1" ]]; then
        printf '\n## %s file coverage diff\n' "${label}"
        cat "${text_report}"
    fi

    return "${status}"
}

python_diff_status=0
c_diff_status=0

run_diff_cover "Python" "${python_coverage_xml}" "python-diff-coverage" || python_diff_status=$?
run_diff_cover "C/C++" "${c_coverage_xml}" "c-diff-coverage" || c_diff_status=$?

ci_log "Checking incremental coverage thresholds."
set +e
python - \
    "${INCREMENTAL_COVERAGE_REPORT_DIR}/python-diff-coverage.json" \
    "${INCREMENTAL_COVERAGE_REPORT_DIR}/c-diff-coverage.json" \
    "${DIFF_COVER_FAIL_UNDER}" <<'PY'
import json
import sys
from pathlib import Path


def load_stats(path):
    report_path = Path(path)
    if not report_path.is_file():
        raise SystemExit(f"Diff coverage JSON not found: {report_path}")
    with report_path.open(encoding="utf-8") as f:
        return json.load(f)


def check(label, path, fail_under):
    stats = load_stats(path)
    total = int(stats.get("total_num_lines") or 0)
    missing = int(stats.get("total_num_violations") or 0)
    percent = float(stats.get("total_percent_covered") or 0.0)
    covered = total - missing

    print(f"{label} incremental coverage:")
    print(f"  line: {percent:.2f}% >= {fail_under:.2f}%")
    print(f"  diff lines: {total}")
    print(f"  covered: {covered}")
    print(f"  missing: {missing}")
    if total == 0:
        print("  note: no lines with coverage information in this diff")

    if percent < fail_under:
        return (
            f"{label} incremental coverage {percent:.2f}% "
            f"is below required {fail_under:.2f}%"
        )
    return None


python_json, c_json, fail_under_arg = sys.argv[1:]
fail_under = float(fail_under_arg)
failures = []
for label, path in (
    ("Python", python_json),
    ("C/C++", c_json),
):
    failure = check(label, path, fail_under)
    if failure:
        failures.append(failure)

if failures:
    print("\nIncremental coverage threshold check failed:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    raise SystemExit(1)
PY
summary_status=$?
set -e

if ((python_diff_status != 0 || c_diff_status != 0 || summary_status != 0)); then
    exit 1
fi

