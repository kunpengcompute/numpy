#!/usr/bin/env bash

set -euo pipefail

ci_repo_root() {
    local script_dir
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

    if command -v git >/dev/null 2>&1; then
        if git -C "${script_dir}" rev-parse --show-toplevel >/dev/null 2>&1; then
            git -C "${script_dir}" rev-parse --show-toplevel
            return 0
        fi
    fi

    # Fallback for non-git environments: .gitcode/pipline lives two levels
    # below the repository root in this copied layout.
    cd -- "${script_dir}/../.." && pwd
}

ci_enter_repo_root() {
    cd -- "$(ci_repo_root)"
}

ci_log() {
    printf '[%s] %s\n' "$(basename "$0")" "$*"
}

ci_require_command() {
    local cmd
    for cmd in "$@"; do
        if ! command -v "${cmd}" >/dev/null 2>&1; then
            printf 'Missing required command: %s\n' "${cmd}" >&2
            return 1
        fi
    done
}

ci_python_deps_mode() {
    local mode="${PY_DEPS_MODE:-check}"
    case "${mode}" in
        check|install)
            printf '%s\n' "${mode}"
            ;;
        *)
            printf 'Unsupported PY_DEPS_MODE: %s (expected check or install)\n' "${mode}" >&2
            return 1
            ;;
    esac
}

ci_should_install_python_deps() {
    [[ "$(ci_python_deps_mode)" == "install" ]]
}

ci_require_python_modules() {
    python - "$@" <<'PY'
import importlib
import sys

missing = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

if missing:
    print("Missing required Python module(s): " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
}

ci_install_python_requirements() {
    local requirements_file="$1"
    if ci_should_install_python_deps; then
        ci_log "Installing Python requirements from ${requirements_file}."
        python -m pip install -r "${requirements_file}"
    else
        ci_log "PY_DEPS_MODE=check, skipping install for ${requirements_file}."
    fi
}

ci_install_python_packages() {
    if ci_should_install_python_deps; then
        ci_log "Installing Python packages: $*"
        python -m pip install "$@"
    else
        ci_log "PY_DEPS_MODE=check, skipping Python package install: $*"
    fi
}

ci_activate_python() {
    local python_bin="${PYTHON_BIN:-python3}"
    local venv_dir="${VENV_DIR:-}"

    ci_python_deps_mode >/dev/null
    ci_require_command "${python_bin}"

    if [[ "${USE_VENV:-0}" == "1" ]]; then
        if [[ -z "${venv_dir}" ]]; then
            printf 'VENV_DIR must be set when USE_VENV=1\n' >&2
            return 1
        fi

        if [[ ! -d "${venv_dir}" || "${RECREATE_VENV:-0}" == "1" ]]; then
            rm -rf -- "${venv_dir}"
            "${python_bin}" -m venv "${venv_dir}"
        fi

        # shellcheck disable=SC1090
        source "${venv_dir}/bin/activate"
    fi

    python -m pip --version >/dev/null
}

ci_detect_pkg_manager() {
    if command -v apt-get >/dev/null 2>&1; then
        printf 'apt-get\n'
        return 0
    fi

    if command -v dnf >/dev/null 2>&1; then
        printf 'dnf\n'
        return 0
    fi

    if command -v yum >/dev/null 2>&1; then
        printf 'yum\n'
        return 0
    fi

    printf 'No supported package manager found. Expected one of: apt-get, dnf, yum\n' >&2
    return 1
}

ci_install_system_packages() {
    if [[ "${SKIP_SYSTEM_DEPS:-1}" == "1" ]]; then
        ci_log "Skipping system package installation."
        return 0
    fi

    local apt_packages="${1:-}"
    local dnf_packages="${2:-${apt_packages}}"
    local yum_packages="${3:-${dnf_packages}}"
    local pkg_manager
    local package_str
    local -a prefix=()
    local -a packages=()

    pkg_manager="$(ci_detect_pkg_manager)"

    if [[ "$(id -u)" -ne 0 ]]; then
        ci_require_command sudo
        prefix=(sudo)
    fi

    case "${pkg_manager}" in
        apt-get)
            package_str="${apt_packages}"
            ci_require_command apt-get
            "${prefix[@]}" apt-get update
            ;;
        dnf)
            package_str="${dnf_packages}"
            ci_require_command dnf
            "${prefix[@]}" dnf makecache
            ;;
        yum)
            package_str="${yum_packages}"
            ci_require_command yum
            "${prefix[@]}" yum makecache
            ;;
        *)
            printf 'Unsupported package manager: %s\n' "${pkg_manager}" >&2
            return 1
            ;;
    esac

    if [[ -z "${package_str}" ]]; then
        printf 'No package list configured for package manager: %s\n' "${pkg_manager}" >&2
        return 1
    fi

    read -r -a packages <<< "${package_str}"
    ci_log "Installing system packages via ${pkg_manager}: ${package_str}"
    "${prefix[@]}" "${pkg_manager}" install -y "${packages[@]}"
}

ci_dump_meson_log() {
    local log_path
    log_path="$(ci_repo_root)/build/meson-logs/meson-log.txt"
    if [[ -f "${log_path}" ]]; then
        printf '\n===== Meson log =====\n' >&2
        cat "${log_path}" >&2
        printf '===== End Meson log =====\n' >&2
    fi
}

ci_prepare_openblas32() {
    ci_require_command spin
    ci_install_python_requirements requirements/ci32_requirements.txt
    if ! ci_should_install_python_deps; then
        ci_require_python_modules scipy_openblas32
    fi
    python -m spin config-openblas --with-scipy-openblas=32
    export PKG_CONFIG_PATH="$(ci_repo_root)/.openblas${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
}

ci_run_via_script_shell() {
    local cmd_file
    cmd_file="$(mktemp)"
    printf '#!/usr/bin/env bash\nset -euo pipefail\n%s\n' "$1" > "${cmd_file}"
    chmod +x "${cmd_file}"

    if command -v script >/dev/null 2>&1; then
        script -q -e -c "bash --noprofile --norc -eo pipefail \"${cmd_file}\"" /dev/null
    else
        bash --noprofile --norc -eo pipefail "${cmd_file}"
    fi

    rm -f -- "${cmd_file}"
}
