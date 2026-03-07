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
│       ├── legacy/
│       ├── run_g1_object_search_demo.ps1
│       ├── run_g1_pointgoal.ps1
│       ├── run_isaac_bridge.ps1
│       ├── run_local_stack.ps1
│       ├── run_live_smoke.ps1
│       ├── run_live_smoke_attach.ps1
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
│   │   ├── isaac_bridge_app.py
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
│   │   ├── isaac_launch_modes.py
│   │   └── live_smoke_runner.py
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
  - shared low-level execution helper for `PlanningSession`, `TrajectoryTracker`, and `ActionStatus` evaluation
- `src/runtime/g1_bridge.py`
  - low-level subgoal executor on top of locomotion and planning session
- `src/runtime/isaac_bridge_runtime.py`
  - standalone Isaac Sim live bridge command source and runtime bootstrap helper
- `src/runtime/live_smoke_runner.py`
  - phase-based live smoke diagnostics, D455 mount verification, and minimal perception->memory ingress checks
- `src/runtime/bootstrap_diagnostics.py`
  - bootstrap phase tracker and diagnostics artifact writer
- `src/runtime/isaac_launch_modes.py`
  - standalone vs attach vs extension mode selection and recommendations
- `src/runtime/isaac_editor_bridge.py`
  - attach-to-running-editor bridge runtime that reuses the live bridge command source inside an existing Kit/Isaac session
- `src/runtime/memory_agent_runtime.py`
  - persistent memory-agent polling loop, periodic diagnostics, and snapshot persistence
- `src/runtime/supervisor.py`
  - consumes tasks, observations, and statuses; emits `ActionCommand`
- `src/apps/runtime_common.py`
  - shared bus/shm/frame-source helpers for local stack and two-process apps
- `src/apps/isaac_bridge_editor_app.py`
  - helper entrypoint for attaching the bridge to an already running Isaac editor/stage
- `src/apps/live_smoke_app.py`
  - dedicated preflight/smoke entrypoint for live bootstrap diagnostics
- `src/adapters/sensors/d455_mount.py`
  - D455 asset resolution, stage mount helper, and prim tree reporting for smoke diagnostics
- `src/inference/detectors`
  - detector backend abstraction, TensorRT capability reporting, YOLOE post-processing, and fallback detector
- `src/perception`
  - detector/tracker/depth projection to `ObsObject`, plus stable person re-id
- `src/memory`
  - structured memory stores, query engine, consolidation, persistence
- `src/services`
  - task orchestration, follow, attention, object recall, semantic consolidation, memory facade
- `src/ipc`
  - message schemas, ZMQ control/telemetry transport, transport health tracking
- `src/adapters/legacy_http` and `src/apps/legacy_http`
  - compatibility-only HTTP path

## Default Execution Path
- Local debug:
  - `apps.local_stack_app`
- Two-process:
  - `apps.memory_agent_app`
  - `apps.isaac_bridge_app`
- Live smoke diagnostics:
  - `apps.live_smoke_app`
- Attach-to-running-editor:
  - `apps.isaac_bridge_editor_app`
- Low-level Isaac/G1 execution:
  - `runtime.g1_bridge`
  - `runtime.isaac_bridge_runtime`

## Detector Path
- Engine discovery starts from `artifacts/models/yoloe-26s-seg-pf.engine`.
- `DetectorRuntimeReport` explains whether TensorRT import, deserialize, binding, and runtime execution are usable.
- If TensorRT load or runtime execution is unavailable, fallback detector remains active.

## Current Limits
- TensorRT execution still depends on a matching engine/runtime/CUDA environment.
- Standalone live smoke depends on Isaac `python.bat`, `SimulationApp`, and a valid D455 asset path; attach mode depends on a running Full App stage.
- Multi-agent fan-out is now retained-replay based on the control plane; targeted routing is still not implemented.
- Legacy HTTP wrappers remain in the tree for compatibility only.
