# Planner Subsystem

- Scope: public planner facade kept for compatibility while planner execution now runs inside the inference stack and task decomposition logic lives under `systems.control.tasking`.
- Package root: `src/systems/planner`
- Public surface:
  - `api/runtime.py`
- Runtime note:
  - planner serving moved to `systems.inference.planner.server`
  - planner tasking moved to `systems.control.tasking`
