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
- `--frame-source auto` stays live-first with synthetic fallback.

## Persistent Memory Agent
```powershell
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --serve --agent-id memory_agent_a
```
- Keeps the structured memory/orchestrator loop alive.
- Periodically republishes diagnostics and snapshots SQLite memory state.

## Two-Process Local Stack
Process 1:
```powershell
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --serve
```

Process 2:
```powershell
.\scripts\powershell\run_isaac_bridge.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --frame-source live --headless --command "아까 봤던 사과를 찾아가"
```
- Control plane: ZMQ `ROUTER/DEALER`
- Telemetry plane: ZMQ `PUB/SUB`
- RGB/depth should use `SharedMemoryRing` in 2-process mode.

## Live Smoke Preflight
```powershell
.\scripts\powershell\run_live_smoke_preflight.ps1
.\scripts\powershell\run_live_smoke_preflight.ps1 -ClearCache -Warmup
```
- Uses Isaac bundled `python.bat`, not system Python.
- Produces a compatibility report with:
  - `isaac_root_found`
  - `isaac_python_found`
  - `experience_found`
  - `assets_root_found`
  - `d455_asset_found`
  - `required_extensions_available`
  - `launch_mode_supported`
  - `editor_assisted_supported`
  - `extension_mode_supported`
  - `warmup_scripts_available`
  - `likely_runtime_mismatch`
  - `recommended_profile`
  - `recommended_launch_mode`

## Standalone Live Smoke
```powershell
.\scripts\powershell\run_live_smoke.ps1 --headless
.\scripts\powershell\run_live_smoke.ps1 --headless --bootstrap-profile minimal_headless_sensor_smoke
.\scripts\powershell\run_live_smoke.ps1 --headless --bootstrap-profile standalone_render_warmup --smoke-target-tier memory
```
- This is the main standalone live bootstrap diagnostic path.
- PowerShell watches the diagnostics JSON and enforces per-phase timeout budgets.
- Key bootstrap phases:
  - `process_start`
  - `isaac_python_env_resolved`
  - `simulation_app_created`
  - `required_extensions_ready`
  - `stage_ready`
  - `stage_opened_or_created`
  - `assets_root_resolved`
  - `d455_asset_resolved`
  - `d455_prim_spawned`
  - `d455_reference_bound`
  - `sensor_wrapper_created`
  - `d455_depth_sensor_initialized`
  - `warmup_frames_started`
  - `warmup_frames_completed`
  - `annotators_ready`
  - `render_products_ready`
  - `first_rgb_frame_ready`
  - `first_depth_frame_ready`
  - `first_nonempty_frame_ready`
  - `first_pose_ready`
  - `perception_ingress_ready`
  - `memory_ingress_ready`
  - `sensor_smoke_pass`
  - `pipeline_smoke_pass`
  - `memory_smoke_pass`
  - `full_smoke_pass`

## Bootstrap Profiles
- `minimal_headless_sensor_smoke`
  - standalone headless
  - short warmup
  - target tier `sensor`
- `standalone_render_warmup`
  - standalone headless
  - heavier render/physics warmup
  - target tier `memory`
- `full_app_editor_assisted`
  - Full App / Kit in-editor execution
  - target tier `memory`
- `extension_in_editor`
  - extension-hosted in-editor execution
  - target tier `memory`

## Smoke Tiers
- `sensor_smoke_pass`
  - app bootstrap
  - D455 asset resolve
  - prim mount
  - sensor init
  - RGB/depth ingress
  - pose or sim time ingress
- `pipeline_smoke_pass`
  - perception pipeline received the frame
  - detector may legally return an empty batch
- `memory_smoke_pass`
  - `MemoryService.update_from_observation()` ran at least once
- Empty scene interpretation:
  - `frame_received=true`
  - `detection_attempted=true`
  - `detections_nonempty=false`
  - `memory_update_called=false`
  means sensor/pipeline ingress worked, but the scene contained no usable detections.

## Editor-Assisted Smoke
```powershell
.\scripts\powershell\run_live_smoke_attach.ps1
```
- Official in-editor launch mode is `editor_assisted`.
- `full_app_attach` remains accepted only as a deprecated alias.
- This is not external process attach.
- `run_live_smoke_attach.ps1` is a compatibility shim:
  - without `-ExtensionMode`, it prints the Script Editor snippet for `apps.editor_smoke_entry.run_editor_smoke(...)`
  - with `-ExtensionMode`, it forwards to the packaged extension path
- Actual `editor_assisted` execution must happen inside a running Isaac Sim Full App / Kit process with an active stage.

## Extension Mode
```powershell
.\scripts\powershell\run_live_smoke_extension.ps1
.\scripts\powershell\run_live_smoke_extension.ps1 -AutoRun -SmokeArgs "--diagnostics-path .\tmp\process_logs\live_smoke\extension.json --smoke-target-tier memory"
```
- Enables the packaged repo extension `isaac.aura.live_smoke`.
- Manual flow:
  1. Launch Isaac Full App with the extension enabled.
  2. Use menu `Isaac Aura > Run Live Smoke`.
- Auto-run flow:
  - set `-AutoRun`
  - optional smoke CLI args are passed through `ISAAC_AURA_LIVE_SMOKE_ARGS`

## Matching Isaac Environment Procedure
1. Run preflight with Isaac `python.bat`.
2. Start with `minimal_headless_sensor_smoke` to prove D455 mount and first-frame ingress.
3. If sensor tier passes, retry with `standalone_render_warmup --smoke-target-tier memory`.
4. If standalone fails before first frame, switch to `editor_assisted`.
5. If you want hot-reload/debugging inside Full App, use `extension_mode`.

## D455 Smoke Expectations
- Expected asset path:
  - `/Isaac/Sensors/Intel/RealSense/rsd455.usd`
- Default mounted prim:
  - `/World/realsense_d455`
- Diagnostics artifacts include:
  - D455 asset resolution
  - mount report
  - stage prim tree
  - enabled extensions
  - sensor init report
  - first frame report
  - smoke metrics
  - compatibility report

## Detector Backend
- Preferred backend order:
  1. `artifacts/models/yoloe-26s-seg-pf.engine`
  2. TensorRT backend if runtime-compatible
  3. color-seg fallback otherwise
- TensorRT mismatch remains a graceful fallback, not a forced workaround.

## Current Limits
- `editor_assisted` and `extension_mode` require in-editor execution. External process attach is not implemented.
- Standalone headless smoke still depends on matching Isaac/Kit/rendering environment.
- Empty-scene memory tier can remain incomplete even when sensor/pipeline tiers pass.
- Multi-agent fan-out is still not targeted routing.
