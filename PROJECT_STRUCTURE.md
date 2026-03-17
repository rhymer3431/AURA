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
├── logs/
├── scripts/
│   └── powershell/
│       ├── run_aura_runtime.ps1
│       ├── run_dashboard.ps1
│       ├── run_dual_server.ps1
│       ├── run_internvla_system2.ps1
│       ├── run_local_stack.ps1
│       ├── run_memory_agent.ps1
│       ├── run_memory_monitor.ps1
│       ├── run_navdp_server.ps1
│       ├── run_system2_optional.ps1
│       └── run_vlm_dual_server.ps1
├── src/
│   ├── adapters/
│   │   └── sensors/
│   │       ├── d455_mount.py
│   │       ├── d455_sensor.py
│   │       ├── frame_source.py
│   │       ├── isaac_bridge_adapter.py
│   │       └── isaac_live_source.py
│   ├── apps/
│   │   ├── deprecated/
│   │   ├── dashboard_backend_app.py
│   │   ├── frame_bridge_app.py
│   │   ├── frame_bridge_editor_app.py
│   │   ├── local_stack_app.py
│   │   ├── memory_agent_app.py
│   │   ├── live_smoke_app.py
│   │   ├── runtime_common.py
│   │   └── webrtc_gateway_app.py
│   ├── common/
│   ├── control/
│   ├── inference/
│   │   ├── detectors/
│   │   │   ├── capabilities.py
│   │   │   └── postprocess/
│   │   ├── navdp/
│   │   ├── trackers/
│   │   └── vlm/
│   ├── modules/
│   ├── mission/
│   ├── planning/
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
│   │   ├── aura_runtime.py
│   │   ├── memory_agent_runtime.py
│   │   ├── navigation_runtime.py
│   │   ├── planning_session.py
│   │   ├── subgoal_executor.py
│   │   └── supervisor.py
│   ├── services/
│   │   ├── attention_service.py
│   │   ├── dual_orchestrator.py
│   │   ├── follow_service.py
│   │   ├── memory_service.py
│   │   ├── mission_manager.py
│   │   ├── object_search_service.py
│   │   ├── planning_coordinator.py
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
- `src/runtime/navigation_runtime.py`
  - canonical main runtime owner coordinating observation, world model, mission, planning, execution, and runtime I/O
- `src/runtime/planning_session.py`
  - planner-owned session facade for point-goal, no-goal, and dual backends
- `src/runtime/subgoal_executor.py`
  - execution backend that turns planner output into locomotion commands
- `src/runtime/aura_runtime.py`
  - deprecated compatibility wrapper for `NavigationRuntime`
- `src/modules/`
  - phase 1 runtime module facades for observation, world model, mission, planning, execution, and runtime I/O
- `src/mission/mission_manager.py`
  - mission-module facade over the legacy `TaskOrchestrator`
- `src/planning/coordinator.py`
  - planning-module facade over the legacy `DualOrchestrator`
- `src/runtime/frame_bridge_runtime.py`
  - internal frame bridge bootstrap for live frame publishing
- `src/runtime/live_smoke_runner.py`
  - deprecated diagnostics runtime pending decommission
- `src/apps/live_smoke_app.py`
  - deprecated diagnostics shim pending decommission
- `src/apps/local_stack_app.py`
  - deprecated single-process shim pending decommission
- `src/apps/frame_bridge_editor_app.py`
  - internal frame bridge attach helper for existing Kit/Isaac sessions
- `src/dashboard_backend/` and `src/webrtc/`
  - supporting dashboard/viewer shell around the canonical runtime

## Default Execution Path
- Canonical:
  - `runtime.navigation_runtime`
- Supporting:
  - `apps.memory_agent_app`
  - `apps.dashboard_backend_app`
  - `apps.webrtc_gateway_app`
- Deprecated / decommission:
  - `apps.local_stack_app`
  - `apps.live_smoke_app`

## Current Limits
- TensorRT execution still depends on a matching engine/runtime/CUDA environment.
- `editor_assisted` and `extension_mode` require in-editor execution; external process attach is not implemented.
- Multi-agent command arbitration is still shared-topic merge, not targeted routing.
- Legacy HTTP wrappers remain in the tree for compatibility only.
