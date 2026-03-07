# Running

## Direct Path
- Direct IPC is the default architecture.
- Legacy HTTP remains compatibility-only.

## Local Stack
```powershell
.\scripts\powershell\run_local_stack.ps1 --command "아까 봤던 사과를 찾아가"
.\scripts\powershell\run_local_stack.ps1 --command "따라와" --scene person --frame-source auto
```
- Uses `InprocBus`.
- Runs detector -> memory -> orchestrator -> `ActionCommand` in one process.
- `--frame-source auto` is the default and uses live-first fallback behavior.

## Memory Agent Loopback
```powershell
.\scripts\powershell\run_memory_agent.ps1 --loopback --command "아까 봤던 사과를 찾아가" --frame-source auto
```
- Useful for validating the bridge/agent IPC flow without a second process.

## Persistent Memory Agent
```powershell
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --serve --agent-id memory_agent_a
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --serve --agent-id memory_agent_b
```
- `--serve` keeps the agent alive and polling.
- The agent republishes diagnostics periodically so a restarted bridge can rediscover it.
- SQLite snapshot persistence runs periodically in serve mode.

## Two-Process Local Stack
Process 1:
```powershell
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561
```

Process 2:
```powershell
.\scripts\powershell\run_isaac_bridge.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --frame-source live --headless --command "아까 봤던 사과를 찾아가"
```

- Small control messages travel over the ZMQ control plane.
- `FrameHeader` and status telemetry travel over the ZMQ telemetry plane.
- RGB/depth should use `SharedMemoryRing` in 2-process mode.
- If ZMQ or shared memory is unavailable, use the in-process loopback mode.
- In live mode the bridge process owns `SimulationApp` directly and runs the low-level `PlanningSession + TrajectoryTracker` executor locally.

## Live Smoke Preflight
```powershell
.\scripts\powershell\run_live_smoke_preflight.ps1
.\scripts\powershell\run_live_smoke_preflight.ps1 -ClearCache -Warmup
```
- Uses Isaac's bundled `python.bat`, not system Python.
- Verifies:
  - Isaac root and `python.bat`
  - optional `clear_caches.bat` / `warmup.bat`
  - Python-side live smoke diagnostics path
  - D455 asset target resolution for `/Isaac/Sensors/Intel/RealSense/rsd455.usd`

## Live Smoke
```powershell
.\scripts\powershell\run_live_smoke.ps1 --headless
.\scripts\powershell\run_live_smoke.ps1 --headless --app-bootstrap-timeout-sec 120 --first-frame-timeout-sec 30
```
- This is the preferred command for diagnosing live bootstrap issues.
- It writes:
  - diagnostics JSON under `tmp/process_logs/live_smoke/`
  - prim tree, extension list, D455 mount report, sensor init report, and first-frame report under the same artifact root
  - stdout/stderr logs under `logs/`
- The launcher watches the diagnostics file and kills the process when the current phase exceeds its own timeout budget.
- Phase order:
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

## Live Smoke Attach / Extension
```powershell
.\scripts\powershell\run_live_smoke_attach.ps1
.\scripts\powershell\run_live_smoke_attach.ps1 -ExtensionMode
```
- Use this when standalone headless bootstrap is the unstable part.
- `full_app_attach`
  - expects a running Isaac Sim Full App / Kit stage
  - reuses that stage instead of creating `SimulationApp`
- `extension_mode`
  - same assumption as attach
  - intended for hot-reload / in-editor debugging
- If no active stage exists, the diagnostics artifact will fail early and recommend a better launch mode.

## Frame Source Modes
```powershell
.\scripts\powershell\run_isaac_bridge.ps1 --frame-source auto
.\scripts\powershell\run_isaac_bridge.ps1 --frame-source live --headless
.\scripts\powershell\run_isaac_bridge.ps1 --frame-source synthetic
.\scripts\powershell\run_isaac_bridge.ps1 --frame-source live --headless --sensor-report-path .\tmp\isaac_live_smoke_report.json
```
- `auto`
  - default
  - tries standalone Isaac bootstrap first
  - emits `RuntimeNotice` and falls back to synthetic if `isaacsim` bootstrap is unavailable
- `live`
  - requires standalone Isaac bootstrap and live RGB/depth capture
  - startup exits non-zero when Isaac bootstrap or live capture is unavailable
- `synthetic`
  - deterministic smoke-test path

## D455 Smoke Expectations
- Asset path:
  - `/Isaac/Sensors/Intel/RealSense/rsd455.usd`
- Default mounted prim:
  - `/World/realsense_d455`
- Successful live smoke means:
  - Isaac app bootstrap succeeded
  - D455 asset resolved
  - D455 prim mounted
  - sensor initialized
  - at least one RGB/depth frame arrived
  - pose or sim time metadata arrived
  - one observation batch reached the local perception/memory ingress path
- Diagnostics separate:
  - `frame received but no detections`
  - `detections produced`
  - `memory updated`

## Editor Attach
```python
from apps.isaac_bridge_editor_app import attach_current_stage

session = attach_current_stage(
    controller=my_controller,
    argv=[
        "--bus", "zmq",
        "--control-endpoint", "tcp://127.0.0.1:5560",
        "--telemetry-endpoint", "tcp://127.0.0.1:5561",
        "--frame-source", "live",
    ],
)
session.tick()
session.close()
```
- This path is for Script Editor or custom extension code inside an already running Kit/Isaac session.
- The host editor owns `SimulationApp`; the attached bridge just reuses the current stage/controller.

## Detector Backend
- Preferred backend order:
  1. `artifacts/models/yoloe-26s-seg-pf.engine`
  2. TensorRT backend if runtime-compatible
  3. color-seg fallback otherwise
- Structured diagnostics are exposed through `DetectorRuntimeReport`.
- In a TensorRT serialization-mismatch environment, the runtime reports the mismatch and falls back cleanly.

## Low-Level G1 Executor
```powershell
.\scripts\powershell\run_g1_pointgoal.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0
.\scripts\powershell\run_g1_pointgoal.ps1 --planner-mode dual --instruction "아까 봤던 사과를 찾아가"
```
- `runtime.g1_bridge` is now a subgoal executor.
- `planning_session.py` uses direct in-process NavDP execution by default.

## Semantic Retrieval Loop
- Task completion creates structured episodes with candidate places, candidate objects, recovery actions, and summary tags.
- `SemanticConsolidationService` converts those episodes into rule-like semantic memory.
- Object recall and follow recovery read those rules back into `WorkingMemory` scoring and `ActionCommand.metadata`.

## Legacy Compatibility
```powershell
.\scripts\powershell\legacy\run_navdp_server.ps1 --port 8888
.\scripts\powershell\legacy\run_vlm_dual_server.ps1 --port 8890 --navdp-url http://127.0.0.1:8888
```

## Current Limits
- TensorRT execution still depends on a matching engine/runtime/CUDA environment.
- Control-plane fan-out is broadcast-based; targeted per-agent routing is still not implemented.
- `apps.memory_agent_app` has a persistent serve mode, but it is still single-process and not supervised by an external service manager.
- Attach/extension live smoke still requires code to execute inside a running Isaac/Kit process with an active stage.
- Standalone headless live smoke can still fail before `SimulationApp` returns; when that happens, use the diagnostics JSON, wrapper summary, and logs to identify the last running phase.
- System2/VLM is optional and not required for the fast path.
