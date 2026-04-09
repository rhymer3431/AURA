# Control Subsystem

- Scope: top-level runtime orchestration, locomotion policy control, operator command ingress, and the main launch surfaces.
- Package root: `src/systems/control`
- Layers:
  - `api`: `play_g1_internvla_navdp.py`, `runtime.py`, `runtime_controller.py`, `runtime_args.py`, `nav_command_api.py`
  - `application`: `entrypoint.py`, `runtime_controller.py`, `runtime_orchestrator.py`
  - `domain`: `constants.py`
  - `infrastructure`: `operator_input.py`, `policy_controller.py`, `policy_session.py`, `training_config.py`
- Entry points:
  - `python -m systems.control.api.play_g1_internvla_navdp`
  - `src/systems/control/bin/run_sim_g1_internvla_navdp_windows.bat`
  - `src/systems/control/bin/send_internvla_nav_command_windows.bat`
