# Runtime Modes

## 1. Local Stack
- Entry: `scripts/powershell/run_local_stack.ps1`
- Module: `apps.local_stack_app`
- Purpose:
  - single-process debug path
  - direct IPC architecture without HTTP
  - detector -> memory -> orchestrator -> command flow in one process
- Default transport:
  - `InprocBus`
- Default frame source:
  - `--frame-source auto`
  - tries live Isaac input first, then falls back to synthetic frames with a runtime notice

## 2. Memory Agent
- Entry: `scripts/powershell/run_memory_agent.ps1`
- Module: `apps.memory_agent_app`
- Purpose:
  - runs structured memory plus task orchestration
  - can loop back locally or run as a persistent ZMQ agent
  - republishes diagnostics so a restarted bridge can re-register the agent
  - periodically persists structured memory snapshots to SQLite in serve mode
- Modes:
  - `--bus inproc --loopback`
  - `--bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --serve`

## 3. Isaac Bridge
- Entry: `scripts/powershell/run_isaac_bridge.ps1`
- Module: `apps.isaac_bridge_app`
- Purpose:
  - in `live` or live-capable `auto`, owns standalone `SimulationApp`
  - publishes `FrameHeader` plus RGB/depth payloads over IPC
  - drains `ActionCommand` and executes low-level NavDP/G1 subgoals locally
  - publishes `ActionStatus`, `RuntimeNotice`, and sensor capability diagnostics
  - can still run loopback without Isaac Sim for smoke checks
- Modes:
  - `--bus inproc --loopback`
  - `--bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --frame-source live --headless`

## 4. Live Smoke Diagnostics
- Entry:
  - `scripts/powershell/run_live_smoke_preflight.ps1`
  - `scripts/powershell/run_live_smoke.ps1`
  - `scripts/powershell/run_live_smoke_attach.ps1`
- Module:
  - `apps.live_smoke_app`
  - `runtime.live_smoke_runner`
- Purpose:
  - standardize live smoke on Windows Isaac `python.bat`
  - separate live bootstrap diagnostics from the main bridge runtime
  - verify D455 asset resolve, mount, sensor initialization, first frame ingress, and minimal perception->memory flow
- Launch modes:
  - `standalone_python`
    - owns `SimulationApp` in a dedicated process
    - best for headless smoke and CI-adjacent reproducibility
  - `full_app_attach`
    - expects an already running Isaac Sim Full App / Kit stage
    - best when standalone bootstrap itself is unstable
  - `extension_mode`
    - same attach assumptions as `full_app_attach`
    - intended for hot-reload / extension debugging inside the editor process
- Phase list:
  - `process_start`
  - `isaac_python_env_resolved`
  - `simulation_app_created`
  - `required_extensions_ready`
  - `stage_ready`
  - `assets_root_resolved`
  - `d455_asset_resolved`
  - `d455_prim_spawned`
  - `d455_depth_sensor_initialized`
  - `render_products_ready`
  - `first_rgb_frame_ready`
  - `first_depth_frame_ready`
  - `first_pose_ready`
  - `observation_batch_processed`
  - `memory_updated`
  - `smoke_pass`
- Artifact outputs:
  - diagnostics JSON in `tmp/process_logs/live_smoke/`
  - prim tree dumps, enabled extension list, D455 mount report, sensor init report, and first-frame report under the same artifact root
  - launcher stdout/stderr logs in `logs/`
- Timeout policy:
  - per-phase budgets for app bootstrap, stage readiness, sensor init, and first frame
  - the PowerShell smoke launcher watches the diagnostics artifact and kills the process when the current phase exceeds its own budget

## 5. Low-Level G1 Executor
- Entry: `scripts/powershell/run_g1_pointgoal.ps1`
- Module: `runtime.g1_bridge`
- Purpose:
  - keeps NavDP + G1 locomotion as low-level execution
  - consumes subgoals from supervisor/task layer
  - uses direct in-process NavDP execution by default

## 6. Legacy Compatibility
- `scripts/powershell/legacy/run_navdp_server.ps1`
- `scripts/powershell/legacy/run_vlm_dual_server.ps1`
- These remain compatibility-only paths and are not the default runtime.

## Detector Backend Priority
1. TensorRT engine discovery at `artifacts/models/yoloe-26s-seg-pf.engine`
2. TensorRT backend if runtime and engine are compatible
3. Color segmentation fallback otherwise

## Frame Source Modes
- `auto`
  - default
  - live-first policy
  - if `isaacsim` standalone bootstrap is unavailable, falls back to synthetic with `RuntimeNotice`
- `live`
  - standalone Isaac Sim only
  - startup fails if bootstrap or camera initialization is unavailable
- `synthetic`
  - deterministic development/smoke-test path

## 3b. Editor Attach Bridge
- Entry:
  - `apps.isaac_bridge_editor_app.attach_current_stage(...)`
- Purpose:
  - attaches the bridge to an already running Kit/Isaac editor session
  - reuses the same live bridge command source without owning `SimulationApp`
  - intended for Script Editor or custom extension integration
- Limits:
  - the host/editor is responsible for providing the active controller and calling `tick()`
  - this path is not yet packaged as a full Omniverse extension

## TensorRT Capability Reporting
- `TensorRtYoloeDetector` now emits a structured capability report.
- The report records:
  - engine presence
  - TensorRT import status
  - engine deserialize status
  - serialization mismatch detection
  - binding metadata availability
  - selected backend and selection reason
- In the current environment, a serialization mismatch still falls back to the color-seg backend.

## Windows Live Smoke Principle
- Live smoke should use Isaac's bundled `python.bat`, not the system Python interpreter.
- `run_live_smoke_preflight.ps1` verifies:
  - Isaac root exists
  - `python.bat` exists
  - optional `clear_caches.bat` and `warmup.bat` availability
  - the Python-side diagnostics path can resolve the D455 asset target
- `run_isaac_bridge.ps1 --frame-source live` now errors out early when Isaac `python.bat` is unavailable instead of silently falling back to system Python.

## D455 Smoke Target
- Expected asset path:
  - `/Isaac/Sensors/Intel/RealSense/rsd455.usd`
- Default smoke mount path:
  - `/World/realsense_d455`
- Smoke diagnostics record:
  - resolved assets root
  - final D455 asset path
  - mounted prim path
  - discovered child prims and depth sensor paths
  - discovered render product paths
  - first-frame timestamps and capture metadata

## Troubleshooting
- If `simulation_app_created` times out:
  - run `run_live_smoke_preflight.ps1`
  - consider `run_live_smoke_attach.ps1` or `--launch-mode full_app_attach`
- If `d455_asset_resolved` fails:
  - check the asset root and `/Isaac/Sensors/Intel/RealSense/rsd455.usd`
- If `render_products_ready` fails in headless standalone:
  - retry `full_app_attach` or `extension_mode`
- If `first_rgb_frame_ready` or `first_depth_frame_ready` times out:
  - inspect `sensor_init_report.json`, `d455_mount_report.json`, and the stage prim tree dump
- `--ClearCache` and `--Warmup` are exposed on the preflight/smoke PowerShell launchers for troubleshooting; they are opt-in, not the default path.

## Current Limits
- Two-process mode now supports multiple agent subscribers on the same bridge, with retained control replay for late joiners.
- Control-plane fan-out is broadcast-based; targeted routing per agent identity is still out of scope.
- `full_app_attach` and `extension_mode` still require the user to run inside an existing Isaac/Kit process with an active stage.
- Standalone live smoke can still fail before `SimulationApp` returns; the new diagnostics path is designed to show which phase was in progress when the watchdog killed it.
- System2/VLM remains optional and is not in the fast path.
