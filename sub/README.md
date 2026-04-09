# Subsystem Architecture

`run_sim_g1_internvla_navdp` now runs on subsystem packages under `src/systems`, with Isaac Sim runtime assembly isolated under `src/simulation`.

Each subsystem is layered as:

- `api`
  - CLI, HTTP, and public facades
- `application`
  - orchestration and use-case services
- `domain`
  - pure models, rules, and algorithms
- `infrastructure`
  - filesystem, subprocess, sensor, and backend adapters

Cross-subsystem imports are restricted to:

- `systems.shared.contracts.*`
- `systems.<subsystem>.api.*`

The runtime subsystems are:

1. `Navigation Subsystem`
   - Owns NavDP client/server, navigation geometry, goal expansion, follower logic, and the backend policy stack.
   - Primary code roots: `src/systems/navigation/api`, `src/systems/navigation/domain`, `src/systems/navigation/infrastructure`.

2. `Inference Subsystem`
   - Owns the InternVLA HTTP server, llama.cpp sidecar process management, session probing, and multimodal response parsing.
   - Primary code roots: `src/systems/inference/api`, `src/systems/inference/infrastructure`.

3. `Perception Subsystem`
   - Owns camera APIs, camera pitch runtime services, camera prim attachment, and sensor capture helpers.
   - Primary code roots: `src/systems/perception/api`, `src/systems/perception/application`, `src/systems/perception/infrastructure`.

4. `World State Subsystem`
   - Owns current runtime state snapshots and shared state contracts across planner, navigation, inference, and control.
   - Primary code roots: `src/systems/world_state/api`.

5. `Planner Subsystem`
   - Owns task-frame normalization, planner HTTP calls, ontology/schema validation, reporting, and subgoal orchestration.
   - Primary code roots: `src/systems/planner/api`, `src/systems/planner/application`, `src/systems/planner/domain`, `src/systems/planner/infrastructure`.

6. `Control Subsystem`
   - Owns operator command ingress, runtime command API, and non-simulation control coordination.
   - Primary code roots: `src/systems/control/api`, `src/systems/control/infrastructure`, `src/systems/control/bin`.

The Isaac Sim runtime package is:

- `Simulation Runtime`
  - Owns the play entrypoint, runtime orchestrator, controller implementations, scene spawn, policy execution, asset resolution, and observation layout.
  - Primary code roots: `src/simulation/api`, `src/simulation/application`, `src/simulation/domain`, `src/simulation/infrastructure`.

Shared DTOs and runtime contracts live in `src/systems/shared/contracts`.

Primary entrypoints are now:

- `python -m systems.control.api.play_g1_internvla_navdp`
- `python -m systems.navigation.api.navdp_server`
- `python -m systems.inference.api.serve_internvla_nav_server`
- `python -m systems.inference.api.check_internvla_session`
- `src/systems/control/bin/run_sim_g1_internvla_navdp_windows.bat`
- `src/systems/navigation/bin/run_navdp_server_windows.bat`
- `src/systems/inference/bin/run_internvla_nav_server_windows.bat`
