# Project Structure

## Directory Tree
```text
.
├── artifacts/
│   ├── indices/
│   └── models/
├── docs/
│   ├── architecture/
│   │   ├── IPC_PROTOCOL.md
│   │   ├── MEMORY_ARCHITECTURE.md
│   │   └── RUNTIME_MODES.md
│   └── refactor_baseline/
├── exts/
│   └── isaac.aura.live_smoke/
├── logs/
├── scripts/
│   └── powershell/
│       ├── legacy/
│       ├── run_g1_object_search_demo.ps1
│       ├── run_g1_pointgoal.ps1
│       ├── run_isaac_bridge.ps1
│       ├── run_local_stack.ps1
│       ├── run_live_smoke.ps1
│       ├── run_live_smoke_attach.ps1
│       ├── run_live_smoke_extension.ps1
│       ├── run_live_smoke_preflight.ps1
│       └── run_memory_agent.ps1
├── src/
│   ├── adapters/
│   │   ├── legacy_http/
│   │   └── sensors/
│   │       ├── d455_mount.py
│   │       ├── d455_sensor.py
│   │       ├── frame_source.py
│   │       ├── isaac_bridge_adapter.py
│   │       └── isaac_live_source.py
│   ├── apps/
│   │   ├── legacy_http/
│   │   ├── editor_smoke_entry.py
│   │   ├── isaac_bridge_app.py
│   │   ├── isaac_bridge_editor_app.py
│   │   ├── live_smoke_app.py
│   │   ├── local_stack_app.py
│   │   ├── memory_agent_app.py
│   │   └── runtime_common.py
│   ├── common/
│   ├── control/
│   ├── inference/
│   │   ├── detectors/
│   │   │   ├── capabilities.py
│   │   │   └── postprocess/
│   │   ├── navdp/
│   │   ├── trackers/
│   │   └── vlm/
│   ├── ipc/
│   │   ├── messages.py
│   │   ├── transport_health.py
│   │   └── zmq_bus.py
│   ├── locomotion/
│   │   └── g1/
│   ├── memory/
│   │   ├── consolidation.py
│   │   ├── models.py
│   │   ├── semantic_store.py
│   │   └── working_memory.py
│   ├── perception/
│   │   ├── person_tracker.py
│   │   ├── pipeline.py
│   │   └── reid_store.py
│   ├── runtime/
│   │   ├── bootstrap_diagnostics.py
│   │   ├── bootstrap_profiles.py
│   │   ├── compatibility_report.py
│   │   ├── isaac_launch_modes.py
│   │   ├── live_smoke_runner.py
│   │   ├── recommendation_engine.py
│   │   └── smoke_result_model.py
│   ├── services/
│   │   ├── attention_service.py
│   │   ├── follow_service.py
│   │   ├── memory_service.py
│   │   ├── object_search_service.py
│   │   ├── semantic_consolidation.py
│   │   └── task_orchestrator.py
│   └── vendor/
├── state/
│   ├── ipc/
│   └── memory/
├── tests/
│   ├── integration/
│   ├── ipc/
│   ├── memory/
│   ├── perception/
│   └── services/
└── tmp/
```

## Responsibilities
- `src/runtime/planning_session.py`
  - direct in-process NavDP facade for point-goal and no-goal execution
- `src/runtime/subgoal_executor.py`
  - shared low-level execution helper for `PlanningSession`, `TrajectoryTracker`, and `ActionStatus`
- `src/runtime/g1_bridge.py`
  - low-level subgoal executor
- `src/runtime/isaac_bridge_runtime.py`
  - standalone Isaac live bridge bootstrap
- `src/runtime/live_smoke_runner.py`
  - phase-based live smoke diagnostics, smoke tier aggregation, and minimal perception/memory ingress validation
- `src/runtime/bootstrap_profiles.py`
  - bootstrap profile selection for standalone/editor/extension smoke paths
- `src/runtime/compatibility_report.py`
  - structured environment compatibility report and recommended launch mode/profile
- `src/runtime/recommendation_engine.py`
  - next-action recommendations from compatibility + failed phase + smoke result
- `src/runtime/smoke_result_model.py`
  - sensor/pipeline/memory tier result model
- `src/apps/live_smoke_app.py`
  - preflight/smoke CLI entrypoint
- `src/apps/editor_smoke_entry.py`
  - official in-editor smoke callable reused by editor-assisted and extension mode
- `src/apps/isaac_bridge_editor_app.py`
  - bridge attach helper for existing Kit/Isaac sessions
- `src/adapters/sensors/d455_mount.py`
  - D455 asset resolution and stage mount helper
- `exts/isaac.aura.live_smoke`
  - packaged Isaac extension with menu/action entry for running live smoke in-editor

## Default Execution Path
- Local debug:
  - `apps.local_stack_app`
- Two-process:
  - `apps.memory_agent_app`
  - `apps.isaac_bridge_app`
- Live smoke diagnostics:
  - `apps.live_smoke_app`
- In-editor smoke:
  - `apps.editor_smoke_entry`
  - `exts/isaac.aura.live_smoke`

## Live Smoke Concepts
- Official launch modes:
  - `standalone_python`
  - `editor_assisted`
  - `extension_mode`
- Deprecated alias:
  - `full_app_attach`
- Smoke tiers:
  - `sensor`
  - `pipeline`
  - `memory`
  - `full`

## Current Limits
- TensorRT execution still depends on a matching engine/runtime/CUDA environment.
- `editor_assisted` and `extension_mode` require in-editor execution; external process attach is not implemented.
- Multi-agent command arbitration is still shared-topic merge, not targeted routing.
- Legacy HTTP wrappers remain in the tree for compatibility only.
