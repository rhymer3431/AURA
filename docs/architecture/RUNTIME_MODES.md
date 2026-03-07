# Runtime Modes

## Default Direct Modes
- `scripts/powershell/run_local_stack.ps1`
  - In-process debug stack.
  - Uses `InprocBus`, `MemoryService`, and `TaskOrchestrator`.
- `scripts/powershell/run_memory_agent.ps1`
  - Starts the structured-memory/task layer without HTTP services.
- `scripts/powershell/run_isaac_bridge.ps1`
  - Intended Isaac-side direct bridge launcher.
  - Uses Isaac Sim Python and the new direct-runtime entrypoint scaffold.

## Low-Level Executor Mode
- `scripts/powershell/run_g1_pointgoal.ps1`
  - Keeps the existing G1 locomotion bridge available.
  - Useful while direct orchestration is being integrated with live Isaac observations.

## Legacy Compatibility Modes
- `scripts/powershell/legacy/run_navdp_server.ps1`
  - Flask NavDP sidecar.
- `scripts/powershell/legacy/run_vlm_dual_server.ps1`
  - Flask dual orchestrator.
- `scripts/powershell/run_navdp_server.ps1`
  - Thin wrapper that forwards to the legacy launcher.
- `scripts/powershell/run_vlm_dual_server.ps1`
  - Thin wrapper that forwards to the legacy launcher.

## Optional System2
- `scripts/powershell/run_system2_optional.ps1`
  - Convenience wrapper around the existing InternVLA launcher.
  - Not required for fast-path direct memory execution.

## Current Boundary
- Direct runtime owns task semantics, memory query, critic/recovery, and IPC.
- Legacy HTTP remains for compatibility and incremental migration.
- NavDP/G1 remains the low-level trajectory executor.
