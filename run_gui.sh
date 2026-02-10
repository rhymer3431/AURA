#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AURA_DIR="$PROJECT_DIR/AURA"
ROS2_DIR="$PROJECT_DIR/ros2_ws_orbslam3"
DASHBOARD_DIR="$PROJECT_DIR/dashboard"

AURA_PORT="${AURA_PORT:-8000}"
AURA_PYTHON="${AURA_PYTHON:-$AURA_DIR/.venv/bin/python}"
YOLO_WEIGHT="${YOLO_WEIGHT:-$AURA_DIR/models/yoloe-26s-seg.pt}"
DASHBOARD_URL="${DASHBOARD_URL:-http://127.0.0.1:${AURA_PORT}/dashboard/}"
DASHBOARD_AUTO_OPEN="${DASHBOARD_AUTO_OPEN:-1}"
DASHBOARD_LAUNCH_MODE="${DASHBOARD_LAUNCH_MODE:-webapp}"
DASHBOARD_FULLSCREEN="${DASHBOARD_FULLSCREEN:-1}"
DASHBOARD_READY_TIMEOUT_SEC="${DASHBOARD_READY_TIMEOUT_SEC:-45}"
DASHBOARD_READY_POLL_SEC="${DASHBOARD_READY_POLL_SEC:-1}"
AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"
AMENT_PYTHON_EXECUTABLE="${AMENT_PYTHON_EXECUTABLE-$(command -v python3)}"

cleanup() {
  local aura_pid="${AURA_PID:-}"
  if [ -z "$aura_pid" ] || ! kill -0 "$aura_pid" >/dev/null 2>&1; then
    return 0
  fi

  # Prefer API shutdown so managed run_pipeline.sh is also terminated.
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -m 2 -X POST "http://127.0.0.1:${AURA_PORT}/system/server/shutdown" >/dev/null 2>&1 || true
    local deadline=$((SECONDS + 6))
    while kill -0 "$aura_pid" >/dev/null 2>&1 && [ "$SECONDS" -lt "$deadline" ]; do
      sleep 0.2
    done
  fi

  if kill -0 "$aura_pid" >/dev/null 2>&1; then
    kill -TERM "$aura_pid" >/dev/null 2>&1 || true
    wait "$aura_pid" 2>/dev/null || true
  fi
}

ensure_dashboard_build() {
  local build_dir="$DASHBOARD_DIR/build"
  if [ -f "$build_dir/index.html" ]; then
    return 0
  fi

  if ! command -v npm >/dev/null 2>&1; then
    echo "[error] dashboard/build missing and npm not found." >&2
    return 1
  fi

  echo "[info] dashboard/build not found. Building dashboard..."
  (
    cd "$DASHBOARD_DIR"
    if [ ! -d node_modules ]; then
      npm ci >/dev/null 2>&1 || npm install >/dev/null 2>&1
    fi
    npm run build >/tmp/aura_dashboard_build.log 2>&1
  ) || {
    echo "[error] dashboard build failed. Check: /tmp/aura_dashboard_build.log" >&2
    return 1
  }
}

start_aura_server() {
  if [ ! -x "$AURA_PYTHON" ]; then
    echo "[error] AURA venv python not found: $AURA_PYTHON" >&2
    echo "[error] Create/activate AURA virtualenv first (expected: $AURA_DIR/.venv)." >&2
    return 1
  fi

  (
    set +u
    if [ -f /opt/ros/humble/setup.bash ]; then
      # shellcheck disable=SC1091
      source /opt/ros/humble/setup.bash
    fi
    if [ -f "$ROS2_DIR/install/setup.bash" ]; then
      # shellcheck disable=SC1091
      source "$ROS2_DIR/install/setup.bash"
    fi
    set -u
    cd "$AURA_DIR"
    export INPUT_SOURCE="${INPUT_SOURCE:-ros2}"
    export PORT="$AURA_PORT"
    export YOLO_WEIGHT
    export YOLO_MODEL="$YOLO_WEIGHT"
    export ENABLE_LLM="${ENABLE_LLM:-0}"
    export AMENT_TRACE_SETUP_FILES
    export AMENT_PYTHON_EXECUTABLE
    "$AURA_PYTHON" -m src.interface.cli.run_server
  ) >/tmp/aura_gui_server.log 2>&1 &
  AURA_PID=$!
}

wait_for_server() {
  local status_url="http://127.0.0.1:${AURA_PORT}/system/pipeline/status"
  local attempt
  for attempt in $(seq 1 30); do
    if ! kill -0 "$AURA_PID" >/dev/null 2>&1; then
      echo "[error] AURA server exited early. Check: /tmp/aura_gui_server.log" >&2
      return 1
    fi
    if command -v curl >/dev/null 2>&1 && curl -fsS -m 2 "$status_url" >/dev/null 2>&1; then
      echo "[info] AURA server ready: http://127.0.0.1:${AURA_PORT}"
      return 0
    fi
    sleep 1
  done

  echo "[error] Timed out waiting for AURA server. Check: /tmp/aura_gui_server.log" >&2
  return 1
}

resolve_dashboard_asset_url() {
  local base_url="$1"
  local dashboard_url="$2"
  local asset_path="$3"

  if [[ "$asset_path" == http://* || "$asset_path" == https://* ]]; then
    printf '%s\n' "$asset_path"
    return 0
  fi

  if [[ "$asset_path" == /* ]]; then
    printf '%s%s\n' "$base_url" "$asset_path"
    return 0
  fi

  asset_path="${asset_path#./}"
  printf '%s%s\n' "$dashboard_url" "$asset_path"
}

wait_for_dashboard_ready() {
  local base_url="http://127.0.0.1:${AURA_PORT}"
  local dashboard_url="${base_url}/dashboard/"
  local timeout_sec="$DASHBOARD_READY_TIMEOUT_SEC"
  local poll_sec="$DASHBOARD_READY_POLL_SEC"
  local deadline=$((SECONDS + timeout_sec))

  if ! command -v curl >/dev/null 2>&1; then
    echo "[warn] curl not found. Skipping dashboard readiness wait."
    return 0
  fi

  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ -n "${AURA_PID:-}" ] && ! kill -0 "$AURA_PID" >/dev/null 2>&1; then
      echo "[warn] AURA server exited before dashboard became ready."
      return 1
    fi

    local dashboard_html
    dashboard_html="$(curl -fsSL -m 2 "$dashboard_url" 2>/dev/null || true)"
    if [ -n "$dashboard_html" ]; then
      local script_path style_path script_url style_url
      script_path="$(printf '%s' "$dashboard_html" | grep -oE 'src="[^"]+\.js"' | head -n 1 | sed -E 's/^src="([^"]+)"$/\1/')"
      style_path="$(printf '%s' "$dashboard_html" | grep -oE 'href="[^"]+\.css"' | head -n 1 | sed -E 's/^href="([^"]+)"$/\1/')"

      if [ -n "$script_path" ]; then
        script_url="$(resolve_dashboard_asset_url "$base_url" "$dashboard_url" "$script_path")"
        if curl -fsS -m 2 "$script_url" >/dev/null 2>&1; then
          if [ -z "$style_path" ]; then
            echo "[info] Dashboard frontend is ready."
            return 0
          fi

          style_url="$(resolve_dashboard_asset_url "$base_url" "$dashboard_url" "$style_path")"
          if curl -fsS -m 2 "$style_url" >/dev/null 2>&1; then
            echo "[info] Dashboard frontend is ready."
            return 0
          fi
        fi
      fi
    fi

    sleep "$poll_sec"
  done

  echo "[warn] Timed out waiting for dashboard readiness (${timeout_sec}s)."
  return 1
}

open_dashboard() {
  if [ "$DASHBOARD_AUTO_OPEN" != "1" ]; then
    echo "[info] Auto-open disabled. Open manually: $DASHBOARD_URL"
    return 0
  fi

  local url="$DASHBOARD_URL"
  local mode="$DASHBOARD_LAUNCH_MODE"
  local app_window_flag=""
  if [ "$mode" = "webview" ]; then
    echo "[warn] DASHBOARD_LAUNCH_MODE=webview is deprecated. Using Chromium app mode instead."
    mode="webapp"
  fi
  if [ "$DASHBOARD_FULLSCREEN" = "1" ]; then
    app_window_flag="--start-fullscreen"
  fi

  # WSL: Windows app mode first.
  if grep -qi "microsoft" /proc/version 2>/dev/null; then
    if command -v cmd.exe >/dev/null 2>&1; then
      if [ "$mode" = "webapp" ]; then
        if cmd.exe /C start "" chrome --app="$url" $app_window_flag >/dev/null 2>&1; then
          echo "[info] Dashboard launched as web app (Chrome): $url"
          return 0
        fi
        if cmd.exe /C start "" msedge --app="$url" $app_window_flag >/dev/null 2>&1; then
          echo "[info] Dashboard launched as web app (Edge/Chromium): $url"
          return 0
        fi
      fi
      if cmd.exe /C start "" "$url" >/dev/null 2>&1; then
        echo "[info] Dashboard opened in browser: $url"
        return 0
      fi
    fi
    echo "[warn] Could not auto-open dashboard on WSL. Open manually: $url"
    return 0
  fi

  # Native Linux: app mode browser first.
  if [ "$mode" = "webapp" ]; then
    local app_browsers=(
      "chromium-browser"
      "chromium"
      "google-chrome-stable"
      "google-chrome"
      "microsoft-edge-stable"
      "microsoft-edge"
      "brave-browser"
    )
    local browser
    for browser in "${app_browsers[@]}"; do
      if command -v "$browser" >/dev/null 2>&1; then
        "$browser" --app="$url" --new-window $app_window_flag >/dev/null 2>&1 &
        echo "[info] Dashboard launched as web app (${browser}): $url"
        return 0
      fi
    done
  fi

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1
    echo "[info] Dashboard opened in browser: $url"
    return 0
  fi

  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1
    echo "[info] Dashboard opened in browser: $url"
    return 0
  fi

  echo "[warn] Could not auto-open browser. Open manually: $url"
}

trap cleanup INT TERM EXIT

if [ ! -d "$AURA_DIR" ]; then
  echo "[error] AURA directory not found: $AURA_DIR" >&2
  exit 1
fi

if [ ! -d "$DASHBOARD_DIR" ]; then
  echo "[error] Dashboard directory not found: $DASHBOARD_DIR" >&2
  exit 1
fi

ensure_dashboard_build
start_aura_server
wait_for_server
if ! wait_for_dashboard_ready; then
  echo "[warn] Launching dashboard despite readiness timeout."
fi
open_dashboard
wait "$AURA_PID"
