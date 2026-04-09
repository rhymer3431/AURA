# World State Subsystem

- Scope: current runtime state ownership and snapshot contracts.
- Package root: `src/systems/world_state`
- Layers:
  - `api`: `runtime_state.py`
- Responsibilities:
  - `PlannerInput`, `CommandState`, `CaptureState`, `System2RuntimeState`
  - `GoalState`, `NavDpState`, `ActionOverrideState`, `LocomotionState`
  - `StatusState`, `NavigationPipelineState`, `TaskExecutionState`
  - goal/state helper functions shared by runtime status assembly
