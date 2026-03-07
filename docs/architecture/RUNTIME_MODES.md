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
  - can loop back locally or poll a real bridge process
- Modes:
  - `--bus inproc --loopback`
  - `--bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561`

## 3. Isaac Bridge
- Entry: `scripts/powershell/run_isaac_bridge.ps1`
- Module: `apps.isaac_bridge_app`
- Purpose:
  - publishes `TaskRequest` and `FrameHeader`
  - drains `ActionCommand`
  - can run loopback without Isaac Sim for smoke checks
  - is the live Isaac frame publishing handoff point
- Modes:
  - `--bus inproc --loopback`
  - `--bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561`

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
  - live-first policy with synthetic fallback and `RuntimeNotice`
- `live`
  - live Isaac only
  - startup fails if the live source is unavailable
- `synthetic`
  - deterministic development/smoke-test path

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
- Two-process mode is presently a single bridge plus single memory-agent topology.
- Live Isaac capture outside an initialized Isaac runtime still falls back to synthetic in `auto` mode.
- System2/VLM remains optional and is not in the fast path.
