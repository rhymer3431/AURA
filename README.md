# AURA System

AURA system owns the backend and runtime services that sit behind the dashboard frontend.

## Runtime layout

`src/systems` contains the subsystem packages:

- `control`
- `inference`
- `navigation`
- `perception`
- `planner`
- `shared/contracts`
- `transport`
- `world_state`

Top-level runtime services live directly under `src`:

- `backend`
- `runtime`

`src/simulation` contains the Isaac Sim host runtime:

- entrypoints
- runtime orchestration
- scene and asset loading
- observation layout
- controller binding

## Canonical launchers

- `scripts/run_system/inference_system_windows.bat`
- `scripts/run_system/navigation_system_windows.bat`
- `scripts/run_system/planner_system_windows.bat`
- `scripts/run_system/control_runtime_windows.bat`
- `scripts/run_system/backend_windows.ps1`
- `scripts/run_system/runtime_windows.ps1` (optional standalone runtime surface)
- `scripts/run_system/dashboard_dev_windows.ps1`

## Python entrypoints

- `python -m systems.inference.api.serve_inference_system`
- `python -m systems.navigation.api.serve_navigation_system`
- `python -m systems.planner.api.serve_planner_system`
- `python -m systems.control.api.play_g1_internvla_navdp`
- `python -m backend.api.serve_backend`
- `python -m runtime.api.serve_runtime` (optional standalone runtime surface)

## Default local bring-up

For normal dashboard work, start only:

1. `scripts/run_system/backend_windows.ps1`
2. the dashboard frontend from `C:\Users\mango\project\AURA\dashboard`

The backend owns runtime lifecycle by default, so the dashboard Start/Stop controls do not require `runtime_windows.ps1`.

Use `runtime_windows.ps1` only when you explicitly want an external runtime control plane. In that mode, point the backend at it with `AURA_RUNTIME_URL` or `--runtime-url`.

## Ports

- Backend: `18095`
- Runtime: `18096`
- Inference system: `15880`
- Planner system: `17881`
- Navigation system: `17882`
- Control runtime: `8892`

## Install

```bash
python -m pip install -e .
```

## Test

```bash
pytest tests/test_backend.py ^
  tests/test_runtime.py ^
  tests/test_planner_tasking.py ^
  tests/test_runtime_planner_status.py ^
  tests/test_target_runtime_entrypoints.py ^
  tests/test_target_runtime_paths.py ^
  tests/test_subsystem_architecture.py ^
  tests/scripts/test_windows_fullstack_launcher.py
```
