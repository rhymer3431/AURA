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

## 4. Low-Level G1 Executor
- Entry: `scripts/powershell/run_g1_pointgoal.ps1`
- Module: `runtime.g1_bridge`
- Purpose:
  - keeps NavDP + G1 locomotion as low-level execution
  - consumes subgoals from supervisor/task layer
  - uses direct in-process NavDP execution by default

## 5. Legacy Compatibility
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

## Current Limits
- Two-process mode now supports multiple agent subscribers on the same bridge, with retained control replay for late joiners.
- Control-plane fan-out is broadcast-based; targeted routing per agent identity is still out of scope.
- Actual live smoke still depends on a working local Isaac Sim installation and valid camera prims in the loaded stage.
- System2/VLM remains optional and is not in the fast path.
