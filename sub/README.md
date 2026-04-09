# Subsystem Architecture

`run_sim_g1_internvla_navdp` now runs on a hard-cut subsystem layout under `src/systems`.

Each subsystem is layered as:

- `api`
  - CLI, HTTP, and public facades
- `application`
  - orchestration and use-case services
- `domain`
  - pure models, rules, and algorithms
- `infrastructure`
  - filesystem, subprocess, Isaac Sim, llama.cpp, and backend adapters

Cross-subsystem imports are restricted to:

- `systems.shared.contracts.*`
- `systems.<subsystem>.api.*`

The five runtime subsystems are:

1. `Navigation Subsystem`
   - Owns NavDP client/server, navigation geometry, goal expansion, follower logic, and the backend policy stack.
   - Primary code roots: `src/systems/navigation/api`, `src/systems/navigation/domain`, `src/systems/navigation/infrastructure`.

2. `Inference Subsystem`
   - Owns the InternVLA HTTP server, llama.cpp sidecar process management, session probing, and multimodal response parsing.
   - Primary code roots: `src/systems/inference/api`, `src/systems/inference/infrastructure`.

3. `World State Subsystem`
   - Owns camera sensing, camera pitch control, observation layout, scene/asset resolution, and world-facing runtime data capture.
   - Primary code roots: `src/systems/world_state/api`, `src/systems/world_state/domain`, `src/systems/world_state/infrastructure`.

4. `Planner Subsystem`
   - Owns task-frame normalization, planner HTTP calls, ontology/schema validation, reporting, and subgoal orchestration.
   - Primary code roots: `src/systems/planner/api`, `src/systems/planner/application`, `src/systems/planner/domain`, `src/systems/planner/infrastructure`.

5. `Control Subsystem`
   - Owns the runtime orchestrator, operator command ingress, locomotion policy binding, and the main simulation launch surface.
   - Primary code roots: `src/systems/control/api`, `src/systems/control/application`, `src/systems/control/domain`, `src/systems/control/infrastructure`, `src/systems/control/bin`.

Shared DTOs and runtime contracts live in `src/systems/shared/contracts`.

Primary entrypoints are now:

- `python -m systems.control.api.play_g1_internvla_navdp`
- `python -m systems.navigation.api.navdp_server`
- `python -m systems.inference.api.serve_internvla_nav_server`
- `python -m systems.inference.api.check_internvla_session`
- `src/systems/control/bin/run_sim_g1_internvla_navdp_windows.bat`
- `src/systems/navigation/bin/run_navdp_server_windows.bat`
- `src/systems/inference/bin/run_internvla_nav_server_windows.bat`
