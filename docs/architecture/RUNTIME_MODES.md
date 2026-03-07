# Runtime Modes

## Default Principle
- Default runtime path is direct IPC plus structured memory.
- Live smoke is a diagnostics path, not the main task runtime.

## 1. Local Stack
- Entry: `scripts/powershell/run_local_stack.ps1`
- Module: `apps.local_stack_app`
- Transport: `InprocBus`
- Frame source: `auto` by default

## 2. Memory Agent
- Entry: `scripts/powershell/run_memory_agent.ps1`
- Module: `apps.memory_agent_app`
- Supports one-shot loopback and persistent `--serve` mode.

## 3. Isaac Bridge
- Entry: `scripts/powershell/run_isaac_bridge.ps1`
- Module: `apps.isaac_bridge_app`
- In live mode this owns standalone `SimulationApp` and executes low-level subgoals locally.

## 4. Live Smoke Launch Modes
- `standalone_python`
  - official headless standalone smoke path
  - uses Isaac `python.bat`
  - best for reproducible bootstrap diagnostics
- `editor_assisted`
  - official in-editor smoke path
  - requires code execution inside a running Isaac Sim Full App / Kit process
  - replaces the old `full_app_attach` naming
- `full_app_attach`
  - deprecated alias for `editor_assisted`
  - kept only for compatibility
- `extension_mode`
  - packaged extension path inside Full App / Kit
  - best for hot-reload and menu/action driven smoke runs

## 5. Bootstrap Profiles
- `minimal_headless_sensor_smoke`
  - short warmup
  - target tier `sensor`
- `standalone_render_warmup`
  - heavier render/physics warmup
  - target tier `memory`
- `full_app_editor_assisted`
  - in-editor Full App profile
  - target tier `memory`
- `extension_in_editor`
  - extension-hosted in-editor profile
  - target tier `memory`

Each profile can control:
- required experience/app flavor
- required extensions
- headless expectation
- render warmup update count
- physics warmup step count
- stage settle policy
- sensor init retry policy
- first-frame timeout

## 6. Smoke Tiers
- `sensor`
  - D455 mount/init plus first frame and pose/sim-time ingress
- `pipeline`
  - frame reached the perception pipeline
  - empty detection batch is still a valid pipeline-tier pass
- `memory`
  - memory ingest/update path ran
- `full`
  - sensor + pipeline + memory all passed

## 7. Compatibility Report
Generated during preflight and smoke startup.

Fields:
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
- `blocking_issues[]`
- `warnings[]`

## 8. Recommendation Strategy
- Early standalone failure:
  - verify `python.bat`, install root, experience path
  - recommend `editor_assisted` if standalone remains unstable
- Asset resolution failure:
  - check assets root / Nucleus mount
- First-frame / annotator failure:
  - recommend `standalone_render_warmup`
  - or switch to `editor_assisted`
- Empty detection batch:
  - interpret as empty-scene pipeline pass unless memory tier is required

## 9. Extension Path
- Extension package lives under `exts/isaac.aura.live_smoke/`.
- It exposes menu/action entry `Isaac Aura > Run Live Smoke`.
- It reuses `apps.editor_smoke_entry.run_extension_smoke`.
- Shutdown cleans up action/menu registrations and temporary path injection.

## 10. Current Limits
- External process attach is not supported.
- `run_live_smoke_attach.ps1` is only a compatibility shim that points users to Script Editor or extension execution.
- Standalone headless success still depends on matching Isaac rendering/runtime environment.
- Live smoke validates ingress and memory wiring, not detector quality tuning.
