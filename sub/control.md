# Control Subsystem

- Scope: operator command ingress, runtime command API, and non-simulation control coordination.
- Package root: `src/systems/control`
- Layers:
  - `api`: `play_g1_internvla_navdp.py`, `runtime.py`, `runtime_controller.py`, `runtime_args.py`, `nav_command_api.py`
  - `infrastructure`: `operator_input.py`
- Entry points:
  - `python -m systems.control.api.play_g1_internvla_navdp`
  - `src/systems/control/bin/run_sim_g1_internvla_navdp_windows.bat`
  - `src/systems/control/bin/send_internvla_nav_command_windows.bat`
