#!/usr/bin/env bash
set -e
# Guard against inherited `set -u` from parent shells.
set +u

# ROS setup scripts may reference this variable even when unset.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"
export AMENT_PYTHON_EXECUTABLE="${AMENT_PYTHON_EXECUTABLE-$(command -v python3)}"

PROJECT_DIR=/home/mangoo/project
HABITAT_DIR=$PROJECT_DIR/habitat-sim
ROS2_DIR=$PROJECT_DIR/ros2_ws_orbslam3
AURA_DIR=$PROJECT_DIR/AURA

HABITAT_STREAM_SCRIPT=$HABITAT_DIR/examples/opencv_agent_viewer.py
HABITAT_PLAYER_VIEWER_SCRIPT=${HABITAT_PLAYER_VIEWER_SCRIPT:-$HABITAT_DIR/examples/viewer.py}
PUBLISHER_SCRIPT=$HABITAT_DIR/examples/ros2_zmq_image_pub.py
YOLOE_NODE_SCRIPT=$ROS2_DIR/src/autonomy_stack/autonomy_stack/yoloe_semantic_node.py
SEMANTIC_FUSION_SCRIPT=$ROS2_DIR/src/autonomy_stack/autonomy_stack/semantic_fusion_node.py
ORB_VOC=$PROJECT_DIR/ORB_SLAM3/Vocabulary/ORBvoc.txt
ORB_CFG=$ROS2_DIR/src/orbslam3_ros2/config/rgb-d/HabitatRGBD.yaml
YOLO_MODEL=${YOLO_MODEL:-$PROJECT_DIR/weights/yoloe-26s-seg.pt}
if [ ! -f "$YOLO_MODEL" ] && [ -f "$AURA_DIR/models/yoloe-26s-seg.pt" ]; then
  YOLO_MODEL="$AURA_DIR/models/yoloe-26s-seg.pt"
fi
if [ ! -f "$YOLO_MODEL" ] && [ -f "$AURA_DIR/yoloe-26s-seg.pt" ]; then
  YOLO_MODEL="$AURA_DIR/yoloe-26s-seg.pt"
fi
if [ ! -f "$YOLO_MODEL" ] && [ -f "$PROJECT_DIR/weights/yoloe-26s-seg-pf.pt" ]; then
  YOLO_MODEL="$PROJECT_DIR/weights/yoloe-26s-seg-pf.pt"
fi
if [ ! -f "$YOLO_MODEL" ] && [ -f "$PROJECT_DIR/weights/yoloe-11s-seg.pt" ]; then
  YOLO_MODEL="$PROJECT_DIR/weights/yoloe-11s-seg.pt"
fi
ENABLE_SEMANTIC=${ENABLE_SEMANTIC:-1}
ENABLE_OCTOMAP=${ENABLE_OCTOMAP:-1}
SEMANTIC_CLOUD_TOPIC=${SEMANTIC_CLOUD_TOPIC:-/semantic_map/semantic_cloud}
SEMANTIC_OCTOMAP_CLOUD_TOPIC=${SEMANTIC_OCTOMAP_CLOUD_TOPIC:-/semantic_map/octomap_cloud}
SEMANTIC_PROJECTED_MAP_TOPIC=${SEMANTIC_PROJECTED_MAP_TOPIC:-/semantic_map/projected_map}
SEMANTIC_UNLABELED_CLASS_ID=${SEMANTIC_UNLABELED_CLASS_ID:-1}
SEMANTIC_POSE_FALLBACK_IDENTITY=${SEMANTIC_POSE_FALLBACK_IDENTITY:-1}
SEMANTIC_ALLOW_STALE_POSE=${SEMANTIC_ALLOW_STALE_POSE:-1}
OCTOMAP_RESOLUTION=${OCTOMAP_RESOLUTION:-0.15}
OCTOMAP_FRAME_ID=${OCTOMAP_FRAME_ID:-map}
OCTOMAP_PUBLISH_PROJECTED_MAP=${OCTOMAP_PUBLISH_PROJECTED_MAP:-1}
ROS_PYTHON=/usr/bin/python3
AURA_PYTHON=${AURA_PYTHON:-$AURA_DIR/.venv/bin/python}
AURA_PORT=${AURA_PORT:-8000}
AURA_ENABLE_STREAM_SERVER=${AURA_ENABLE_STREAM_SERVER:-1}
DASHBOARD_AUTO_OPEN=${DASHBOARD_AUTO_OPEN:-1}
DASHBOARD_LAUNCH_MODE=${DASHBOARD_LAUNCH_MODE:-webapp}
DASHBOARD_FULLSCREEN=${DASHBOARD_FULLSCREEN:-1}
DASHBOARD_READY_TIMEOUT_SEC=${DASHBOARD_READY_TIMEOUT_SEC:-45}
DASHBOARD_READY_POLL_SEC=${DASHBOARD_READY_POLL_SEC:-1}
ENABLE_LLM=${ENABLE_LLM:-0}
ROS_IMAGE_TOPIC=${ROS_IMAGE_TOPIC:-/camera/rgb}
ROS_SLAM_POSE_TOPIC=${ROS_SLAM_POSE_TOPIC:-/orbslam/pose}
ROS_SEMANTIC_PROJECTED_MAP_TOPIC=${ROS_SEMANTIC_PROJECTED_MAP_TOPIC:-/semantic_map/projected_map}
ROS_SEMANTIC_OCTOMAP_CLOUD_TOPIC=${ROS_SEMANTIC_OCTOMAP_CLOUD_TOPIC:-/semantic_map/octomap_cloud}
ROS_SLAM_MAP_POINTS_TOPIC=${ROS_SLAM_MAP_POINTS_TOPIC:-/orbslam/map_points}
ROS_MAP_TOPIC=${ROS_MAP_TOPIC:-/map}
ROS_ODOM_TOPIC=${ROS_ODOM_TOPIC:-/odom}
USE_NAV2=${USE_NAV2:-1}
NAV2_PARAMS_FILE=${NAV2_PARAMS_FILE:-$ROS2_DIR/src/autonomy_stack/config/nav2_orbslam.yaml}
NAV2_AUTO_EXPLORE=${NAV2_AUTO_EXPLORE:-1}
FRONTIER_PARAMS_FILE=${FRONTIER_PARAMS_FILE:-$ROS2_DIR/src/autonomy_stack/config/frontier_explorer.yaml}
HABITAT_MANUAL_ONLY=${HABITAT_MANUAL_ONLY:-0}
HABITAT_ACTIONS=${HABITAT_ACTIONS:-turn_right,move_forward,turn_left,move_forward}
HABITAT_CONTROL_HOST=${HABITAT_CONTROL_HOST:-127.0.0.1}
HABITAT_CONTROL_PORT=${HABITAT_CONTROL_PORT:-8766}
HABITAT_ADD_PLAYER_AGENT=${HABITAT_ADD_PLAYER_AGENT:-0}
HABITAT_PLAYER_VIEWER_BACKEND=${HABITAT_PLAYER_VIEWER_BACKEND:-habitat}
HABITAT_ROBOT_ASSET_PATH=${HABITAT_ROBOT_ASSET_PATH:-$HABITAT_DIR/data/objects/robots/robot.glb}
HABITAT_ROBOT_ASSET_SCALE=${HABITAT_ROBOT_ASSET_SCALE:-0.35}
HABITAT_ROBOT_ASSET_Y_OFFSET=${HABITAT_ROBOT_ASSET_Y_OFFSET:-0.0}
HABITAT_SCENE_DATASET=${HABITAT_SCENE_DATASET:-$HABITAT_DIR/data/replica_cad/replicaCAD.scene_dataset_config.json}
HABITAT_SCENE=${HABITAT_SCENE:-}

if [ -z "$HABITAT_SCENE" ]; then
  case "$HABITAT_SCENE_DATASET" in
    *"/scene_datasets/modern_apartment/modern_apartment.scene_dataset_config.json")
      HABITAT_SCENE="modern_apartment"
      ;;
    *"/data/replica_cad_custom/replicaCAD_custom.scene_dataset_config.json"|\
    *"/data/versioned_data/replica_cad_dataset_custom/replicaCAD_custom.scene_dataset_config.json")
      HABITAT_SCENE="multiroom_compound"
      ;;
    *)
      HABITAT_SCENE="apt_1"
      ;;
  esac
fi

if [ ! -f "$HABITAT_SCENE_DATASET" ]; then
  echo "[error] Habitat dataset config not found: $HABITAT_SCENE_DATASET" >&2
  exit 1
fi

echo "[info] Habitat dataset: $HABITAT_SCENE_DATASET"
echo "[info] Habitat scene: $HABITAT_SCENE"
HABITAT_GUI_ENABLED=0
if [ "$HABITAT_ADD_PLAYER_AGENT" = "1" ]; then
  if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    HABITAT_GUI_ENABLED=1
  fi
fi
HABITAT_USE_NATIVE_PLAYER_VIEWER=0
if [ "$HABITAT_ADD_PLAYER_AGENT" = "1" ] && [ "$HABITAT_GUI_ENABLED" = "1" ]; then
  if [ "$HABITAT_PLAYER_VIEWER_BACKEND" = "habitat" ]; then
    if [ -f "$HABITAT_PLAYER_VIEWER_SCRIPT" ]; then
      HABITAT_USE_NATIVE_PLAYER_VIEWER=1
    else
      echo "[warn] HABITAT_PLAYER_VIEWER_SCRIPT not found: $HABITAT_PLAYER_VIEWER_SCRIPT (falling back to OpenCV player view)"
    fi
  fi
fi
PIPELINE_CONTROLLER_PID=${PIPELINE_CONTROLLER_PID:-$BASHPID}
CLEANUP_STARTED=0

ORB_SLAM3_VISUALIZATION=${ORB_SLAM3_VISUALIZATION:-0}
HABITAT_SLEEP_SEC=${HABITAT_SLEEP_SEC:-0.01}
HABITAT_TURN_SLEEP_SEC=${HABITAT_TURN_SLEEP_SEC:-0}

HABITAT_ZMQ="ipc:///tmp/habitat_zmq"

launch_dashboard() {
  if [ "$DASHBOARD_AUTO_OPEN" != "1" ]; then
    return 0
  fi

  local url="http://127.0.0.1:${AURA_PORT}/dashboard/"
  local mode="$DASHBOARD_LAUNCH_MODE"
  local app_window_flag=""
  if [ "$mode" = "webview" ]; then
    echo "[warn] DASHBOARD_LAUNCH_MODE=webview is deprecated. Using Chromium app mode instead."
    mode="webapp"
  fi
  if [ "$DASHBOARD_FULLSCREEN" = "1" ]; then
    app_window_flag="--start-fullscreen"
  fi

  # WSL: Windows 앱 모드 실행 우선
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

  # Native Linux: 앱 모드 브라우저 우선
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
        DASHBOARD_UI_PID=$!
        echo "[info] Dashboard launched as web app (${browser}): $url"
        return 0
      fi
    done
  fi

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 &
    echo "[info] Dashboard opened in browser: $url"
    return 0
  fi

  echo "[warn] Could not auto-open dashboard. Open manually: $url"
}

ensure_dashboard_build() {
  local dashboard_dir="$PROJECT_DIR/dashboard"
  local build_dir="$dashboard_dir/build"

  if [ -d "$build_dir" ] && [ -f "$build_dir/index.html" ]; then
    return 0
  fi

  if ! command -v npm >/dev/null 2>&1; then
    echo "[warn] npm not found. Cannot build dashboard automatically."
    return 1
  fi

  echo "[info] dashboard/build not found. Building dashboard..."
  (
    cd "$dashboard_dir"
    if [ ! -d node_modules ]; then
      npm ci >/dev/null 2>&1 || npm install >/dev/null 2>&1
    fi
    npm run build >/tmp/aura_dashboard_build.log 2>&1
  ) || {
    echo "[warn] dashboard build failed. check log: /tmp/aura_dashboard_build.log"
    return 1
  }

  return 0
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

cleanup() {
  if [ "$CLEANUP_STARTED" = "1" ]; then
    return 0
  fi
  CLEANUP_STARTED=1

  local process_vars=(
    DASHBOARD_UI_PID
    HABITAT_PID
    SLAM_PID
    PUB_PID
    SEMDET_PID
    SEMFUS_PID
    NAV_PID
    MAP_PID
    TFBRIDGE_PID
    EXPLORE_PID
    OCTOMAP_PID
    AURA_PID
    PLAYER_VIEWER_PID
  )
  local running_pids=()
  local var_name
  local pid

  for var_name in "${process_vars[@]}"; do
    pid="${!var_name:-}"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      running_pids+=("$pid")
    fi
  done

  if [ "${#running_pids[@]}" -eq 0 ]; then
    return 0
  fi

  echo "[info] Shutting down pipeline processes (${#running_pids[@]})..."
  for pid in "${running_pids[@]}"; do
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done

  local deadline=$((SECONDS + 8))
  local still_running=("${running_pids[@]}")
  while [ "${#still_running[@]}" -gt 0 ] && [ "$SECONDS" -lt "$deadline" ]; do
    local next_round=()
    for pid in "${still_running[@]}"; do
      if kill -0 "$pid" >/dev/null 2>&1; then
        next_round+=("$pid")
      fi
    done
    still_running=("${next_round[@]}")
    if [ "${#still_running[@]}" -gt 0 ]; then
      sleep 0.2
    fi
  done

  if [ "${#still_running[@]}" -gt 0 ]; then
    for pid in "${still_running[@]}"; do
      kill -KILL "$pid" >/dev/null 2>&1 || true
    done
    echo "[warn] Forced kill for remaining processes: ${still_running[*]}"
  fi
}

trap cleanup INT TERM EXIT

(
  if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda deactivate
  fi
  source /opt/ros/humble/setup.bash
  "$ROS_PYTHON" "$PUBLISHER_SCRIPT"
) &
PUB_PID=$!

(
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate habitat
  cmd=(
    python "$HABITAT_STREAM_SCRIPT"
    --dataset "$HABITAT_SCENE_DATASET"
    --scene "$HABITAT_SCENE"
    --zmq-address "$HABITAT_ZMQ"
    --zmq-topic habitat/rgb
    --zmq-imu-topic habitat/imu
    --send-depth
    --send-imu
    --zmq-rgb-encoding bgr8
    --width 640 --height 480
    --jpeg-quality 80
    --sleep-sec "$HABITAT_SLEEP_SEC"
    --turn-sleep-sec "$HABITAT_TURN_SLEEP_SEC"
    --actions "$HABITAT_ACTIONS"
    --control-host "$HABITAT_CONTROL_HOST"
    --control-port "$HABITAT_CONTROL_PORT"
  )
  if [ "$HABITAT_MANUAL_ONLY" = "1" ]; then
    cmd+=(--manual-only)
  fi
  if [ "$HABITAT_ADD_PLAYER_AGENT" = "1" ]; then
    cmd+=(--add-player-agent)
    if [ "$HABITAT_USE_NATIVE_PLAYER_VIEWER" = "1" ]; then
      cmd+=(--hide-player-view)
    fi
  fi
  if [ -n "$HABITAT_ROBOT_ASSET_PATH" ]; then
    if [ -f "$HABITAT_ROBOT_ASSET_PATH" ]; then
      cmd+=(--robot-asset "$HABITAT_ROBOT_ASSET_PATH")
      cmd+=(--robot-asset-scale "$HABITAT_ROBOT_ASSET_SCALE")
      cmd+=(--robot-asset-y-offset "$HABITAT_ROBOT_ASSET_Y_OFFSET")
    else
      echo "[warn] Habitat robot asset not found: $HABITAT_ROBOT_ASSET_PATH (robot avatar disabled)"
    fi
  fi
  if [ "$HABITAT_GUI_ENABLED" != "1" ]; then
    cmd+=(--no-display)
  fi
  "${cmd[@]}"
) &
HABITAT_PID=$!

if [ "$HABITAT_USE_NATIVE_PLAYER_VIEWER" = "1" ]; then
  (
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate habitat
    python "$HABITAT_PLAYER_VIEWER_SCRIPT" \
      --dataset "$HABITAT_SCENE_DATASET" \
      --scene "$HABITAT_SCENE" \
      --width 640 \
      --height 480 \
      --control-host "$HABITAT_CONTROL_HOST" \
      --control-port "$HABITAT_CONTROL_PORT" \
      --control-target player
  ) &
  PLAYER_VIEWER_PID=$!
fi

sleep 1

if [ "$HABITAT_GUI_ENABLED" = "1" ]; then
  if [ "$HABITAT_USE_NATIVE_PLAYER_VIEWER" = "1" ]; then
    echo "[info] Habitat GUI enabled (native Habitat player viewer mode)."
  else
    echo "[info] Habitat GUI enabled (player agent mode)."
  fi
else
  if [ "$HABITAT_ADD_PLAYER_AGENT" = "1" ]; then
    echo "[warn] HABITAT_ADD_PLAYER_AGENT=1 but no DISPLAY/WAYLAND_DISPLAY detected. Running headless."
  fi
  echo "[info] Habitat/ORB native viewer windows are disabled."
fi

if [ "$AURA_ENABLE_STREAM_SERVER" = "1" ]; then
  if [ ! -x "$AURA_PYTHON" ]; then
    echo "[warn] AURA python not found: $AURA_PYTHON (stream server skipped)"
  else
    (
      if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
        conda deactivate
      fi
      source /opt/ros/humble/setup.bash
      source "$ROS2_DIR/install/setup.bash"
      cd "$AURA_DIR"
      export INPUT_SOURCE=ros2
      export ROS_IMAGE_TOPIC
      export ROS_SLAM_POSE_TOPIC
      export ROS_SEMANTIC_PROJECTED_MAP_TOPIC
      export ROS_SEMANTIC_OCTOMAP_CLOUD_TOPIC
      export PORT="$AURA_PORT"
      export ENABLE_LLM
      export AURA_PIPELINE_CONTROLLER_PID="$PIPELINE_CONTROLLER_PID"
      "$AURA_PYTHON" -m src.interface.cli.run_server
    ) &
    AURA_PID=$!
    echo "[info] AURA stream server: http://127.0.0.1:${AURA_PORT}"
    if ensure_dashboard_build; then
      echo "[info] Dashboard: http://127.0.0.1:${AURA_PORT}/dashboard/"
      if ! wait_for_dashboard_ready; then
        echo "[warn] Launching dashboard despite readiness timeout."
      fi
      launch_dashboard
    else
      echo "[warn] Dashboard launch skipped due to build issue."
    fi
  fi
fi

if [ "$ENABLE_SEMANTIC" = "1" ]; then
  if [ ! -f "$YOLO_MODEL" ]; then
    for candidate in \
      "$AURA_DIR/models/yoloe-26s-seg.pt" \
      "$AURA_DIR/yoloe-26s-seg.pt" \
      "$PROJECT_DIR/weights/yoloe-26s-seg.pt" \
      "$PROJECT_DIR/weights/yoloe-26s-seg-pf.pt" \
      "$PROJECT_DIR/weights/yoloe-11s-seg.pt"; do
      if [ -f "$candidate" ]; then
        YOLO_MODEL="$candidate"
        break
      fi
    done
  fi

  if [ ! -f "$YOLO_MODEL" ]; then
    echo "[warn] YOLO model not found: $YOLO_MODEL (semantic pipeline skipped)"
  else
    echo "[info] Semantic detector model: $YOLO_MODEL"
    if [ -x "$AURA_PYTHON" ] && "$AURA_PYTHON" -c "import ultralytics" >/dev/null 2>&1; then
      (
        source /opt/ros/humble/setup.bash
        source "$ROS2_DIR/install/setup.bash"
        "$AURA_PYTHON" "$YOLOE_NODE_SCRIPT" --ros-args -p model_path:="$YOLO_MODEL"
      ) &
      SEMDET_PID=$!
    elif "$ROS_PYTHON" -c "import ultralytics" >/dev/null 2>&1; then
      (
        source /opt/ros/humble/setup.bash
        source "$ROS2_DIR/install/setup.bash"
        "$ROS_PYTHON" "$YOLOE_NODE_SCRIPT" --ros-args -p model_path:="$YOLO_MODEL"
      ) &
      SEMDET_PID=$!
    else
      echo "[warn] ultralytics not installed (AURA/ROS python). semantic detector skipped."
    fi

    if [ -n "${SEMDET_PID:-}" ]; then
      (
        source /opt/ros/humble/setup.bash
        source "$ROS2_DIR/install/setup.bash"
        "$ROS_PYTHON" "$SEMANTIC_FUSION_SCRIPT" --ros-args \
          -p semantic_cloud_topic:="$SEMANTIC_CLOUD_TOPIC" \
          -p octomap_cloud_topic:="$SEMANTIC_OCTOMAP_CLOUD_TOPIC" \
          -p projected_map_topic:="$SEMANTIC_PROJECTED_MAP_TOPIC" \
          -p unlabeled_class_id:="$SEMANTIC_UNLABELED_CLASS_ID" \
          -p pose_fallback_identity:="$([[ "$SEMANTIC_POSE_FALLBACK_IDENTITY" = "1" ]] && echo true || echo false)" \
          -p allow_stale_pose:="$([[ "$SEMANTIC_ALLOW_STALE_POSE" = "1" ]] && echo true || echo false)" \
          -p publish_projected_map:="$([[ "$OCTOMAP_PUBLISH_PROJECTED_MAP" = "1" ]] && echo true || echo false)"
      ) &
      SEMFUS_PID=$!

      if [ "$ENABLE_OCTOMAP" = "1" ]; then
        if (source /opt/ros/humble/setup.bash && ros2 pkg prefix octomap_server >/dev/null 2>&1); then
          (
            source /opt/ros/humble/setup.bash
            source "$ROS2_DIR/install/setup.bash"
            ros2 run octomap_server octomap_server_node --ros-args \
              -p resolution:="$OCTOMAP_RESOLUTION" \
              -p frame_id:="$OCTOMAP_FRAME_ID" \
              -r /cloud_in:="$SEMANTIC_OCTOMAP_CLOUD_TOPIC"
          ) &
          OCTOMAP_PID=$!
          echo "[info] octomap_server started (cloud=$SEMANTIC_OCTOMAP_CLOUD_TOPIC)"
        else
          echo "[warn] octomap_server package not found. semantic octomap cloud will still be published."
        fi
      fi
    fi
  fi
fi

(
  if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda deactivate
  fi
  cd "$ROS2_DIR"
  source install/setup.bash
  export ORB_SLAM3_VISUALIZATION
  ros2 run orbslam3 rgbd "$ORB_VOC" "$ORB_CFG"
) &
SLAM_PID=$!

if [ "$USE_NAV2" = "1" ]; then
  if ! (source /opt/ros/humble/setup.bash && ros2 pkg prefix nav2_bringup >/dev/null 2>&1); then
    echo "[warn] nav2_bringup 패키지를 찾을 수 없어 기존 navigator로 fallback합니다."
    USE_NAV2=0
  elif [ ! -f "$NAV2_PARAMS_FILE" ]; then
    echo "[warn] Nav2 params 파일이 없습니다: $NAV2_PARAMS_FILE (기존 navigator로 fallback)"
    USE_NAV2=0
  fi
fi

if [ "$USE_NAV2" = "1" ]; then
  echo "[info] Nav2 mode enabled."

  (
    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
      source ~/miniconda3/etc/profile.d/conda.sh
      conda deactivate
    fi
    source /opt/ros/humble/setup.bash
    "$ROS_PYTHON" -c "import sys; sys.path.insert(0, '$ROS2_DIR/src/autonomy_stack'); from autonomy_stack.pose_tf_bridge_node import main; main()" --ros-args \
      -p pose_topic:="$ROS_SLAM_POSE_TOPIC" \
      -p odom_topic:="$ROS_ODOM_TOPIC" \
      -p map_frame:=map \
      -p odom_frame:=odom \
      -p base_frame:=base_link \
      -p publish_map_to_odom:=true \
      -p flatten_to_2d:=true
  ) &
  TFBRIDGE_PID=$!

  (
    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
      source ~/miniconda3/etc/profile.d/conda.sh
      conda deactivate
    fi
    source /opt/ros/humble/setup.bash
    "$ROS_PYTHON" -c "import sys; sys.path.insert(0, '$ROS2_DIR/src/autonomy_stack'); from autonomy_stack.sparse_map_occupancy_node import main; main()" --ros-args \
      -p map_points_topic:="$ROS_SLAM_MAP_POINTS_TOPIC" \
      -p pose_topic:="$ROS_SLAM_POSE_TOPIC" \
      -p map_topic:="$ROS_MAP_TOPIC" \
      -p map_frame:=map
  ) &
  MAP_PID=$!

  (
    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
      source ~/miniconda3/etc/profile.d/conda.sh
      conda deactivate
    fi
    source /opt/ros/humble/setup.bash
    source "$ROS2_DIR/install/setup.bash"
    ros2 launch nav2_bringup navigation_launch.py use_sim_time:=false params_file:="$NAV2_PARAMS_FILE"
  ) &
  NAV_PID=$!

  if [ "$NAV2_AUTO_EXPLORE" = "1" ]; then
    if [ ! -f "$FRONTIER_PARAMS_FILE" ]; then
      echo "[warn] Frontier params 파일이 없습니다: $FRONTIER_PARAMS_FILE (auto explore skipped)"
    else
      (
        if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
          source ~/miniconda3/etc/profile.d/conda.sh
          conda deactivate
        fi
        source /opt/ros/humble/setup.bash
        # nav2 lifecycle activation 대기 후 탐사 시작
        sleep 8
        "$ROS_PYTHON" -c "import sys; sys.path.insert(0, '$ROS2_DIR/src/autonomy_stack'); from autonomy_stack.frontier_explorer_node import main; main()" --ros-args \
          --params-file "$FRONTIER_PARAMS_FILE" \
          -p map_topic:="$ROS_MAP_TOPIC" \
          -p pose_topic:="$ROS_SLAM_POSE_TOPIC" \
          -p action_name:=/navigate_to_pose \
          -p map_frame:=map
      ) &
      EXPLORE_PID=$!
      echo "[info] Nav2 frontier auto-explore enabled."
    fi
  fi
else
  echo "[info] Legacy navigator mode enabled."
  (
    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
      source ~/miniconda3/etc/profile.d/conda.sh
      conda deactivate
    fi
    source /opt/ros/humble/setup.bash
    "$ROS_PYTHON" -c "import sys; sys.path.insert(0, '$ROS2_DIR/src/autonomy_stack'); from autonomy_stack.navigator_node import main; main()" --ros-args \
      --params-file "$ROS2_DIR/src/autonomy_stack/config/autonomy.yaml" \
      -p topics.rgb:=/camera/rgb \
      -p topics.depth:=/camera/depth \
      -p topics.rgb_info:=/camera/rgb/camera_info \
      -p topics.depth_info:=/camera/depth/camera_info \
      -p tf.use_pose_topic:=true \
      -p tf.pose_topic:="$ROS_SLAM_POSE_TOPIC"
  ) &
  NAV_PID=$!
fi

PIDS=("$HABITAT_PID" "$PUB_PID" "$SLAM_PID" "$NAV_PID")
if [ -n "${MAP_PID:-}" ]; then
  PIDS+=("$MAP_PID")
fi
if [ -n "${TFBRIDGE_PID:-}" ]; then
  PIDS+=("$TFBRIDGE_PID")
fi
if [ -n "${EXPLORE_PID:-}" ]; then
  PIDS+=("$EXPLORE_PID")
fi
if [ -n "${SEMDET_PID:-}" ]; then
  PIDS+=("$SEMDET_PID")
fi
if [ -n "${SEMFUS_PID:-}" ]; then
  PIDS+=("$SEMFUS_PID")
fi
if [ -n "${AURA_PID:-}" ]; then
  PIDS+=("$AURA_PID")
fi
if [ -n "${OCTOMAP_PID:-}" ]; then
  PIDS+=("$OCTOMAP_PID")
fi
wait "${PIDS[@]}"
