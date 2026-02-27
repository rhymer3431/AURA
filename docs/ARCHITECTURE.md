# Windows Native Isaac Sim(4.2) + ROS2 + G1 Agent Architecture

## 1) Process layout

- **A. Isaac Sim Runner** (`apps/isaacsim_runner/isaac_runner.py`)
  - Loads `g1/g1_d455.usd` as-is (no transform edits in code).
  - Headless-first execution.
  - Publishes RGB/Depth + TF/joint_states/clock in mock ROS2 mode when `rclpy` is available.
  - Native mode configures Isaac Sim ROS2 bridge graphs for:
    - `/clock`, `/tf`, `/{namespace}/joint_states`
    - `/{namespace}/cmd/joint_commands` -> articulation controller
    - `/{namespace}/camera/{color,depth}` publishers

- **B. Agent Runtime** (`apps/agent_runtime/main.py`)
  - Async orchestrator for perception, SLAM confidence, scene memory, planner client, executor, Nav2 adapter, manipulation adapter, VRAM guard.
  - Mock-first but interface-compatible with real TensorRT engines and ROS2/Nav2.

- **C. Planner Server** (`apps/services/planner_server/server.py`)
  - Local `/plan` API endpoint.
  - Returns JSON-only plan schema.
  - Mock planner enabled by default; Nanbeige INT4 hookup marked as TODO.

## 2) Agent module graph

- **Perception (YOLOE TRT wrapper)**
  - `modules/perception_yoloe_trt.py`
  - TensorRT engine load + CUDA buffers + NMS postprocess.
  - optional ROS2 camera subscriber (`/{namespace}/camera/color/image_raw` or compressed).
  - queue length 1~2 with frame drop.
  - lightweight per-label target tracker (EMA-smoothed center).
  - warmup at startup.

- **SLAM monitor**
  - `modules/slam_monitor.py`
  - computes `C_loc` in `[0,1]` from covariance/quality heuristic.
  - hysteresis thresholds `T_low` / `T_high`.
  - emits mode transitions:
    - `exploration`: low confidence.
    - `localization`: confidence recovered for hold-time.

- **Scene memory STM/LTM**
  - `modules/memory.py`
  - map-anchored object entries.
  - importance-based STM→LTM promotion.
  - API:
    - `get_object_pose(class_or_id)`
    - `set_start_pose(pose)`, `get_start_pose()`
    - `update_from_detection(detections, robot_pose)`

- **Task planner client**
  - `modules/planner_client.py`
  - sends `{user_command, world_state}` to planner server.
  - strict JSON plan handling.
  - fallback stub if planner unavailable.

- **Task executor**
  - `modules/task_executor.py`
  - skills: `locate`, `navigate`, `pick`, `return`, `inspect`, `fetch`, `look_at`
  - retry policy and failure reasons.
  - subscribes to SLAM mode event to pause/resume current task.
  - runs exploration behavior while paused.

- **Look-at controller**
  - `modules/look_at_controller.py`
  - persistent target tracking state: `idle`, `tracking`, `target_lost`, `stopped`
  - P-controller on image error `(cx, cy) -> yaw/pitch`
  - deadband, rate-limit, smoothing, target-lost timeout handling

- **Navigation adapter**
  - `modules/nav2_client.py`
  - TODO: real `nav2_msgs/action/NavigateToPose`.
  - mock returns categorized fail reasons (`NO_PATH`, `TIMEOUT`, `POSE_UNCERTAIN`).

- **Manipulation adapter (GR00T TRT wrapper)**
  - `modules/manipulation_groot_trt.py`
  - supports `gr00t_policy_server` backend (ZeroMQ + msgpack), `trt_engine` placeholder, and mock fallback.
  - loads modality metadata from `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` (`processor_config.json`, `statistics.json`).
  - routes predicted actions via G1 action adapter (`log` or ROS2 topic mode).

- **VRAM guard**
  - `modules/vram_guard.py`
  - NVML polling (`pynvml`), fallback simulation mode.
  - degrade policy:
    1. planner throttling/offload
    2. YOLO frequency down + model variant fallback
    3. Isaac rendering minimization TODO hook

## 3) Runtime strategy

- Warmup heavy engines at startup.
- Keep sensor/control loops free from planner LLM latency.
- Prefer latest-state processing: stale frames are dropped.
- Keep render products minimal (RGB + Depth only).
