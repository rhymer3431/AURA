# Control Subsystem

- Scope: Isaac Sim runtime control, runtime task execution, runtime control HTTP API, and operator input handling.
- Package root: `src/systems/control`
- Modules:
  - `api`: `play_g1_internvla_navdp.py`, `runtime.py`, `runtime_controller.py`, `runtime_args.py`
  - `runtime`: runtime controller and entrypoint facades
  - `tasking`: task-frame planning, subgoal orchestration, and reporting
  - `operator_input.py`
  - `runtime_control_api.py`
- Entry points:
  - `python -m systems.control.api.play_g1_internvla_navdp`
  - `scripts/run_system/control_runtime_windows.bat`
