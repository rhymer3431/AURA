# AURA

AURA is a subsystem-oriented robot runtime for Isaac Sim.

The active codebase now lives in two source roots only:

- `src/systems`
- `src/simulation`

## Runtime layout

`src/systems` contains the runtime subsystems:

- `control`
- `navigation`
- `inference`
- `planner`
- `perception`
- `world_state`
- `transport`
- `shared/contracts`

`src/simulation` contains the Isaac Sim host runtime:

- entrypoints
- runtime orchestration
- scene and asset loading
- observation layout
- controller binding

## Active launchers

Primary launcher:

- `src/systems/control/bin/run_sim_g1_internvla_navdp_windows.bat`

Supporting launchers:

- `src/systems/navigation/bin/run_navdp_server_windows.bat`
- `src/systems/inference/bin/run_internvla_nav_server_windows.bat`
- `scripts/serve_planner_qwen3_nothink.ps1`

Python module entrypoints:

- `python -m systems.control.api.play_g1_internvla_navdp`
- `python -m systems.navigation.api.navdp_server`
- `python -m systems.inference.api.serve_internvla_nav_server`
- `python -m systems.inference.api.check_internvla_session`

## Install

```bash
python -m pip install -e .
```

## Test

Run the active runtime-focused suite with:

```bash
pytest tests/transport ^
  tests/test_planner_tasking.py ^
  tests/test_runtime_planner_status.py ^
  tests/test_target_runtime_entrypoints.py ^
  tests/test_target_runtime_paths.py ^
  tests/test_subsystem_architecture.py ^
  tests/scripts/test_windows_fullstack_launcher.py
```
