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
├── experiments/
│   └── launchers/
├── logs/
├── scripts/
│   └── powershell/
│       ├── legacy/
│       ├── run_g1_object_search_demo.ps1
│       ├── run_g1_pointgoal.ps1
│       ├── run_isaac_bridge.ps1
│       ├── run_local_stack.ps1
│       ├── run_memory_agent.ps1
│       ├── run_navdp_server.ps1
│       ├── run_system2_optional.ps1
│       └── run_vlm_dual_server.ps1
├── src/
│   ├── adapters/
│   │   ├── legacy_http/
│   │   └── sensors/
│   ├── apps/
│   │   ├── legacy_http/
│   │   ├── isaac_bridge_app.py
│   │   ├── local_stack_app.py
│   │   └── memory_agent_app.py
│   ├── common/
│   │   ├── config/
│   │   ├── schemas/
│   │   ├── cv2_compat.py
│   │   ├── geometry.py
│   │   └── scene.py
│   ├── control/
│   ├── inference/
│   │   ├── detectors/
│   │   ├── navdp/
│   │   ├── trackers/
│   │   └── vlm/
│   ├── ipc/
│   ├── locomotion/
│   │   └── g1/
│   ├── memory/
│   ├── perception/
│   ├── runtime/
│   ├── services/
│   └── vendor/
├── state/
│   ├── ipc/
│   └── memory/
│       ├── snapshots/
│       └── vector/
├── tests/
│   ├── integration/
│   ├── ipc/
│   ├── memory/
│   └── services/
└── tmp/
    └── process_logs/
```

## Folder Responsibilities
- `src/adapters/sensors`: Isaac/D455-facing sensor and bridge adapters.
- `src/adapters/legacy_http`: HTTP compatibility clients retained for NavDP/VLM sidecars.
- `src/apps/isaac_bridge_app.py`: direct Isaac bridge entry scaffold.
- `src/apps/memory_agent_app.py`: structured-memory/task-agent entry scaffold.
- `src/apps/local_stack_app.py`: in-process debug stack using the new direct IPC abstractions.
- `src/apps/legacy_http`: Flask compatibility apps isolated from the direct runtime path.
- `src/control`: behavior FSM, critic, recovery, async planners, and trajectory tracking.
- `src/ipc`: bus abstraction, message dataclasses, codec, ZMQ skeleton, and shared-memory ring.
- `src/memory`: spatial/temporal/episodic/semantic/working memory plus query and persistence helpers.
- `src/perception`: observation fusion, object mapping, speaker events, person tracking, and ReID skeletons.
- `src/runtime`: direct runtime supervisor, planning session, Isaac runtime scaffold, and compatibility bridge code.
- `src/services`: task orchestration, intent parsing, attention/follow/object-search services, plus legacy dual orchestrator compatibility.
- `scripts/powershell/legacy`: canonical legacy HTTP launchers.
- `state/memory`: runtime-generated SQLite, vector state, and snapshots.
- `state/ipc`: runtime-created IPC artifacts.

## Default Runtime Path
- Direct path: `run_local_stack.ps1` or `run_isaac_bridge.ps1` -> `apps.local_stack_app` / `apps.isaac_bridge_app`
- Legacy compatibility: `run_navdp_server.ps1`, `run_vlm_dual_server.ps1`
- Low-level G1 executor remains `run_g1_pointgoal.ps1`

## Key Modules
- `src/runtime/planning_session.py`: low-level subgoal execution session for pointgoal/nogoal actions.
- `src/runtime/supervisor.py`: direct runtime coordinator for task requests, observations, and action commands.
- `src/services/task_orchestrator.py`: behavior-state orchestration for attention, follow, recall, local search, and recovery.
- `src/services/memory_service.py`: facade over spatial/temporal/episodic/semantic/working memory.
- `src/ipc/messages.py`: `FrameHeader`, `ActionCommand`, `ActionStatus`, `TaskRequest`.
- `src/adapters/sensors/isaac_bridge_adapter.py`: bus-facing Isaac bridge adapter.
