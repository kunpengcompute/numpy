#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

ci_enter_repo_root

: "${VENV_DIR:=.venv-internal-ci/incremental_coverage}"
: "${COMPARE_BRANCH:=origin/main}"
: "${DIFF_COVER_FAIL_UNDER:=0}"

coverage_xml="build/coverage/coverage.xml"

ci_activate_python

if [[ ! -f "${coverage_xml}" ]]; then
    if [[ "${RUN_FULL_JOB_IF_MISSING:-0}" == "1" ]]; then
        ci_log "Coverage XML missing, running full coverage job first."
        "${SCRIPT_DIR}/full.sh"
    else
        printf 'Coverage XML not found: %s\n' "${coverage_xml}" >&2
        printf 'Run tools/ci/linux-arm/full.sh first or set RUN_FULL_JOB_IF_MISSING=1.\n' >&2
        exit 1
    fi
fi

ci_install_python_packages diff-cover
if ! ci_should_install_python_deps; then
    ci_require_command diff-cover
fi

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

ci_log "Computing incremental coverage."
diff-cover "${coverage_xml}" --compare-branch "${COMPARE_BRANCH}" --fail-under "${DIFF_COVER_FAIL_UNDER}"
