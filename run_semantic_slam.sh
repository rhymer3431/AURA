#!/usr/bin/env bash
set -euo pipefail

# ROS setup scripts expect this variable to exist when nounset(-u) is enabled.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"
export AMENT_PYTHON_EXECUTABLE="${AMENT_PYTHON_EXECUTABLE-$(command -v python3)}"

PROJECT_DIR=${PROJECT_DIR:-/home/mangoo/project}
ROS2_DIR=${ROS2_DIR:-$PROJECT_DIR/ros2_ws_orbslam3}
AURA_DIR=${AURA_DIR:-$PROJECT_DIR/AURA}

ORB_VOC=${ORB_VOC:-$PROJECT_DIR/ORB_SLAM3/Vocabulary/ORBvoc.txt}
ORB_CFG=${ORB_CFG:-$ROS2_DIR/src/orbslam3_ros2/config/rgb-d/HabitatRGBD.yaml}
RUN_ORBSLAM=${RUN_ORBSLAM:-1}

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

SEMANTIC_TOPIC=${SEMANTIC_TOPIC:-/semantic/label}
OVERLAY_TOPIC=${OVERLAY_TOPIC:-/semantic/overlay}
CLASS_MAP_TOPIC=${CLASS_MAP_TOPIC:-/semantic/class_map}
MARKER_TOPIC=${MARKER_TOPIC:-/semantic_map/markers}
SEMANTIC_CLOUD_TOPIC=${SEMANTIC_CLOUD_TOPIC:-/semantic_map/semantic_cloud}
OCTOMAP_CLOUD_TOPIC=${OCTOMAP_CLOUD_TOPIC:-/semantic_map/octomap_cloud}
PROJECTED_MAP_TOPIC=${PROJECTED_MAP_TOPIC:-/semantic_map/projected_map}

RGB_TOPIC=${RGB_TOPIC:-/camera/rgb}
DEPTH_TOPIC=${DEPTH_TOPIC:-/camera/depth}
RGB_INFO_TOPIC=${RGB_INFO_TOPIC:-/camera/rgb/camera_info}
POSE_TOPIC=${POSE_TOPIC:-/orbslam/pose}

ENABLE_OCTOMAP=${ENABLE_OCTOMAP:-1}
OCTOMAP_RESOLUTION=${OCTOMAP_RESOLUTION:-0.15}
OCTOMAP_FRAME_ID=${OCTOMAP_FRAME_ID:-map}
OCTOMAP_PUBLISH_PROJECTED_MAP=${OCTOMAP_PUBLISH_PROJECTED_MAP:-1}
SEMANTIC_UNLABELED_CLASS_ID=${SEMANTIC_UNLABELED_CLASS_ID:-1}
SEMANTIC_POSE_FALLBACK_IDENTITY=${SEMANTIC_POSE_FALLBACK_IDENTITY:-1}
SEMANTIC_ALLOW_STALE_POSE=${SEMANTIC_ALLOW_STALE_POSE:-1}

YOLO_DEVICE=${YOLO_DEVICE:-0}
YOLO_CLASSES=${YOLO_CLASSES:-}
YOLO_DISABLE_OPEN_VOCAB=${YOLO_DISABLE_OPEN_VOCAB:-0}

if [[ "$YOLO_DEVICE" =~ ^[0-9]+$ ]]; then
  YOLO_DEVICE="cuda:$YOLO_DEVICE"
fi

ROS_PYTHON=${ROS_PYTHON:-/usr/bin/python3}
AURA_PYTHON=${AURA_PYTHON:-$AURA_DIR/.venv/bin/python}

if [ ! -f "$YOLO_MODEL" ]; then
  echo "[error] YOLO model not found: $YOLO_MODEL"
  exit 1
fi

if [ ! -f "$ORB_VOC" ] && [ "$RUN_ORBSLAM" = "1" ]; then
  echo "[error] ORB vocabulary not found: $ORB_VOC"
  exit 1
fi

if [ ! -f "$ORB_CFG" ] && [ "$RUN_ORBSLAM" = "1" ]; then
  echo "[error] ORB config not found: $ORB_CFG"
  exit 1
fi

if [ -x "$AURA_PYTHON" ] && "$AURA_PYTHON" -c "import ultralytics" >/dev/null 2>&1; then
  DETECTOR_PYTHON="$AURA_PYTHON"
elif "$ROS_PYTHON" -c "import ultralytics" >/dev/null 2>&1; then
  DETECTOR_PYTHON="$ROS_PYTHON"
else
  echo "[error] ultralytics is not installed in AURA or ROS python."
  exit 1
fi

cleanup() {
  if [ -n "${SEMDET_PID:-}" ]; then
    kill "$SEMDET_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${SEMFUS_PID:-}" ]; then
    kill "$SEMFUS_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${OCTOMAP_PID:-}" ]; then
    kill "$OCTOMAP_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${SLAM_PID:-}" ]; then
    kill "$SLAM_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup INT TERM EXIT

echo "[info] semantic detector model: $YOLO_MODEL"
echo "[info] detector python: $DETECTOR_PYTHON"

(
  set +u
  source /opt/ros/humble/setup.bash
  source "$ROS2_DIR/install/setup.bash"
  set -u
  export PROJECT_DIR

  args=(
    "$ROS2_DIR/src/autonomy_stack/autonomy_stack/yoloe_semantic_node.py"
    --ros-args
    -p rgb_topic:="$RGB_TOPIC"
    -p semantic_topic:="$SEMANTIC_TOPIC"
    -p overlay_topic:="$OVERLAY_TOPIC"
    -p class_map_topic:="$CLASS_MAP_TOPIC"
    -p model_path:="$YOLO_MODEL"
    -p device:="$YOLO_DEVICE"
  )
  if [ -n "$YOLO_CLASSES" ]; then
    args+=(-p classes:="$YOLO_CLASSES")
  fi
  if [ "$YOLO_DISABLE_OPEN_VOCAB" = "1" ]; then
    args+=(-p disable_open_vocab:=true)
  fi
  "$DETECTOR_PYTHON" "${args[@]}"
) &
SEMDET_PID=$!

(
  set +u
  source /opt/ros/humble/setup.bash
  source "$ROS2_DIR/install/setup.bash"
  set -u
  "$ROS_PYTHON" "$ROS2_DIR/src/autonomy_stack/autonomy_stack/semantic_fusion_node.py" --ros-args \
    -p semantic_topic:="$SEMANTIC_TOPIC" \
    -p depth_topic:="$DEPTH_TOPIC" \
    -p rgb_info_topic:="$RGB_INFO_TOPIC" \
    -p pose_topic:="$POSE_TOPIC" \
    -p marker_topic:="$MARKER_TOPIC" \
    -p semantic_cloud_topic:="$SEMANTIC_CLOUD_TOPIC" \
    -p octomap_cloud_topic:="$OCTOMAP_CLOUD_TOPIC" \
    -p projected_map_topic:="$PROJECTED_MAP_TOPIC" \
    -p unlabeled_class_id:="$SEMANTIC_UNLABELED_CLASS_ID" \
    -p pose_fallback_identity:="$([[ "$SEMANTIC_POSE_FALLBACK_IDENTITY" = "1" ]] && echo true || echo false)" \
    -p allow_stale_pose:="$([[ "$SEMANTIC_ALLOW_STALE_POSE" = "1" ]] && echo true || echo false)" \
    -p publish_projected_map:="$([[ "$OCTOMAP_PUBLISH_PROJECTED_MAP" = "1" ]] && echo true || echo false)"
) &
SEMFUS_PID=$!

if [ "$ENABLE_OCTOMAP" = "1" ]; then
  if (
    set +u
    source /opt/ros/humble/setup.bash
    ros2 pkg prefix octomap_server >/dev/null 2>&1
  ); then
    (
      set +u
      source /opt/ros/humble/setup.bash
      source "$ROS2_DIR/install/setup.bash"
      set -u
      ros2 run octomap_server octomap_server_node --ros-args \
        -p resolution:="$OCTOMAP_RESOLUTION" \
        -p frame_id:="$OCTOMAP_FRAME_ID" \
        -r /cloud_in:="$OCTOMAP_CLOUD_TOPIC"
    ) &
    OCTOMAP_PID=$!
    echo "[info] octomap_server started. cloud_in=$OCTOMAP_CLOUD_TOPIC"
  else
    echo "[warn] octomap_server package is not installed; only semantic octomap cloud will be published."
  fi
fi

if [ "$RUN_ORBSLAM" = "1" ]; then
  (
    set +u
    source /opt/ros/humble/setup.bash
    source "$ROS2_DIR/install/setup.bash"
    set -u
    ros2 run orbslam3 rgbd "$ORB_VOC" "$ORB_CFG"
  ) &
  SLAM_PID=$!
fi

echo "[info] semantic SLAM nodes running."
echo "[info] topics: $SEMANTIC_TOPIC, $OVERLAY_TOPIC, $MARKER_TOPIC, $OCTOMAP_CLOUD_TOPIC"
wait
