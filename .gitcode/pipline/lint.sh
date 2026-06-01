#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

ci_enter_repo_root

: "${VENV_DIR:=.venv-internal-ci/lint}"

ci_activate_python

ci_install_python_requirements requirements/linter_requirements.txt
if ! ci_should_install_python_deps; then
    ci_require_command ruff
    ci_require_python_modules git
fi

ci_log "Running Ruff-based lint checks."
python tools/linter.py

ci_log "Checking Python.h include order."
python tools/check_python_h_first.py
