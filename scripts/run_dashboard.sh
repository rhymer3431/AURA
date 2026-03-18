#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DASHBOARD_DIR="${REPO_DIR}/dashboard"
SRC_DIR="${REPO_DIR}/src"

PYTHON_BIN="${AURA_DASHBOARD_PYTHON_CMD:-${AURA_DASHBOARD_PYTHON_EXE:-python3}}"
NPM_BIN="${AURA_DASHBOARD_NPM_CMD:-npm}"
CARGO_BIN="${AURA_DASHBOARD_CARGO_CMD:-cargo}"
ENTRY_MODULE="apps.dashboard_backend_app"
BACKEND_PID=""

log() {
  printf '[AURA_DASHBOARD] %s\n' "$*"
}

die() {
  log "$*"
  exit 1
}

cleanup() {
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    log "stopping backend pid=${BACKEND_PID}"
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "required command not found: ${cmd}"
}

test_python_modules() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("aiohttp", "aiortc", "av", "zmq") if importlib.util.find_spec(name) is None]
sys.exit(0 if not missing else 1)
PY
}

install_python_modules() {
  log "installing missing backend Python modules: aiohttp aiortc av pyzmq"
  "${PYTHON_BIN}" -m pip install aiohttp aiortc av pyzmq
}

wait_http_ready() {
  local url="$1"
  local timeout_sec="$2"
  local deadline=$((SECONDS + timeout_sec))

  while (( SECONDS < deadline )); do
    if [[ -n "${BACKEND_PID}" ]] && ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
      return 1
    fi
    if "${PYTHON_BIN}" - "$url" <<'PY' >/dev/null 2>&1; then
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=2) as response:
    if 200 <= response.status < 300:
        raise SystemExit(0)
raise SystemExit(1)
PY
      return 0
    fi
    sleep 0.5
  done

  return 1
}

resolve_backend_health_url() {
  local host="127.0.0.1"
  local port="8095"
  local args=("$@")
  local index=0

  while (( index < ${#args[@]} )); do
    case "${args[index]}" in
      --host)
        if (( index + 1 < ${#args[@]} )); then
          host="${args[index + 1]}"
          ((index += 2))
          continue
        fi
        ;;
      --host=*)
        host="${args[index]#--host=}"
        ;;
      --port)
        if (( index + 1 < ${#args[@]} )); then
          port="${args[index + 1]}"
          ((index += 2))
          continue
        fi
        ;;
      --port=*)
        port="${args[index]#--port=}"
        ;;
    esac
    ((index += 1))
  done

  if [[ "${host}" == "0.0.0.0" || "${host}" == "localhost" || "${host}" == *:* ]]; then
    host="127.0.0.1"
  fi

  printf 'http://%s:%s/api/bootstrap\n' "${host}" "${port}"
}

[[ -d "${DASHBOARD_DIR}" ]] || die "dashboard directory not found: ${DASHBOARD_DIR}"
[[ -f "${DASHBOARD_DIR}/package.json" ]] || die "dashboard/package.json not found"

require_cmd "${PYTHON_BIN}"
require_cmd "${NPM_BIN}"
require_cmd "${CARGO_BIN}"

if ! test_python_modules; then
  install_python_modules
fi

if [[ ! -d "${DASHBOARD_DIR}/node_modules" ]]; then
  log "installing dashboard node_modules"
  (
    cd "${DASHBOARD_DIR}"
    "${NPM_BIN}" install
  )
fi

"${CARGO_BIN}" -V >/dev/null

BACKEND_HEALTH_URL="$(resolve_backend_health_url "$@")"

log "starting dashboard backend at ${BACKEND_HEALTH_URL%/api/bootstrap}"
(
  cd "${REPO_DIR}"
  export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  exec "${PYTHON_BIN}" -m "${ENTRY_MODULE}" "$@"
) &
BACKEND_PID=$!

log "backend pid=${BACKEND_PID}"
log "waiting for backend readiness at ${BACKEND_HEALTH_URL}"
if ! wait_http_ready "${BACKEND_HEALTH_URL}" 20; then
  die "dashboard backend did not become ready at ${BACKEND_HEALTH_URL}"
fi

log "backend ready"
log "starting Tauri dashboard"
(
  cd "${DASHBOARD_DIR}"
  exec "${NPM_BIN}" run tauri:dev
)
