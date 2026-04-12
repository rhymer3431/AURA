# Subsystem Architecture

`run_sim_g1_internvla_navdp` now runs on subsystem packages under `src/systems`, with Isaac Sim runtime assembly isolated under `src/simulation`.

Each subsystem keeps `api` as the public facade and organizes internal code directly under subsystem-owned modules such as `runtime`, `tasking`, `stack`, `backend`, and `sources`.

Cross-subsystem imports are restricted to:

- `systems.shared.contracts.*`
- `systems.<subsystem>.api.*`

The runtime subsystems are:

1. `Navigation Subsystem`
   - Owns navigation geometry, goal expansion, follower logic, and the NavDP client surface.
   - Primary code roots: `src/systems/navigation/api`, `src/systems/navigation/geometry.py`, `src/systems/navigation/goals.py`, `src/systems/navigation/follower.py`, `src/systems/navigation/client.py`.

2. `Inference Subsystem`
   - Owns the managed inference stack for NavDP, InternVLA System2, planner serving, child-process supervision, and health aggregation.
   - Primary code roots: `src/systems/inference/api`, `src/systems/inference/navdp`, `src/systems/inference/system2`, `src/systems/inference/planner`, `src/systems/inference/stack`.

3. `Memory Subsystem`
   - Owns runtime short-term frame history and exposes STM views that inference consumers can reuse without duplicating frame caches.
   - Primary code roots: `src/systems/memory/api`, `src/systems/memory/stm.py`.

4. `Backend Service`
   - Owns the aiohttp backend, session orchestration, SSE state broadcasting, log aggregation, occupancy metadata, and WebRTC signaling proxy.
   - Primary code roots: `src/backend`, `src/backend/api`, `src/backend/sources`.

5. `Perception Subsystem`
   - Owns camera APIs, camera pitch runtime services, camera prim attachment, and sensor capture helpers.
   - Primary code roots: `src/systems/perception/api`, `src/systems/perception/application`, `src/systems/perception/infrastructure`.

6. `World State Subsystem`
   - Owns current runtime state snapshots and shared state contracts across planner, navigation, inference, and control.
   - Primary code roots: `src/systems/world_state/api`.

7. `Planner Subsystem`
   - Keeps the public planner facade while planner execution is served from the inference stack and task decomposition logic lives in `systems.control.tasking`.
   - Primary code roots: `src/systems/planner/api`, `src/systems/control/tasking`, `src/systems/inference/planner`.

8. `Control Subsystem`
   - Owns operator command ingress, runtime task execution, and the runtime control API.
   - Primary code roots: `src/systems/control/api`, `src/systems/control/runtime`, `src/systems/control/tasking`.

9. `Transport Subsystem`
   - Owns runtime message contracts, in-process and ZMQ buses, shared-memory frame transport, and transport health tracking.
   - Primary code roots: `src/systems/transport`, `src/systems/transport/bus`.

The Isaac Sim runtime package is:

- `Simulation Runtime`
  - Owns the play entrypoint, runtime orchestrator, controller implementations, scene spawn, policy execution, asset resolution, and observation layout.
  - Primary code roots: `src/simulation/api`, `src/simulation/application`, `src/simulation/domain`, `src/simulation/infrastructure`.

Shared DTOs and runtime contracts live in `src/systems/shared/contracts`.

Primary entrypoints are now:

- `python -m systems.control.api.play_g1_internvla_navdp`
- `python -m systems.inference.api.serve_inference_stack`
- `python -m backend.api.serve_backend`
- `python -m runtime.api.serve_runtime` (optional standalone runtime surface)
- `scripts/run_system/control_runtime_windows.bat`
- `scripts/run_system/inference_stack_windows.bat`
- `scripts/run_system/backend_windows.ps1`
- `scripts/run_system/runtime_windows.ps1` (optional standalone runtime surface)
