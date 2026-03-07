# Project Structure

## Directory Tree
```text
.
├── artifacts/
│   └── models/
├── docs/
│   └── refactor_baseline/
├── experiments/
│   └── launchers/
├── logs/
├── scripts/
│   └── powershell/
├── src/
│   ├── adapters/
│   ├── apps/
│   ├── common/
│   ├── control/
│   ├── inference/
│   ├── locomotion/
│   │   └── g1/
│   ├── runtime/
│   ├── services/
│   └── vendor/
├── navdp/
├── navdp_sidecar/
├── g1_play/
├── tests/
└── tmp/
    └── process_logs/
```

## Folder Responsibilities
- `src/adapters`: NavDP HTTP, dual-system HTTP, D455 capture, and external system boundaries.
- `src/apps`: Flask app factories and executable server modules.
- `src/common`: shared geometry, transforms, scene helpers, and other low-level utilities.
- `src/control`: planners, async planner workers, trajectory tracking, and controller-side coordination.
- `src/inference`: policy model code and inference-only logic.
- `src/locomotion`: ONNX locomotion runtime, locomotion controller code, and tightly coupled G1 assets.
- `src/runtime`: bridge orchestration, runtime sessions, CLI contracts, and high-level execution flow.
- `src/services`: service-level request handling and orchestration logic.
- `src/vendor`: bundled third-party code, isolated from first-party modules.
- `navdp`, `navdp_sidecar`, `g1_play`: compatibility shims only. They preserve legacy imports and entrypoints but do not own canonical business logic.

## Entrypoints
- `run_g1_pointgoal.ps1`: root compatibility launcher.
- `scripts/powershell/run_g1_pointgoal.ps1`: canonical bridge launcher for `runtime.g1_bridge`.
- `run_navdp_server.ps1`: root compatibility launcher.
- `scripts/powershell/run_navdp_server.ps1`: canonical NavDP server launcher for `apps.navdp_server_app`.
- `run_vlm_dual_server.ps1`: root compatibility launcher.
- `scripts/powershell/run_vlm_dual_server.ps1`: canonical dual orchestrator launcher for `apps.dual_server_app`.
- `play_g1_keyboard_onnx.py`: compatibility shim to `locomotion.entrypoint`.
- `python -m navdp.g1_bridge`: preserved public compatibility entrypoint backed by `runtime.g1_bridge`.

## Key Runtime Modules
- `src/runtime/g1_bridge.py`: bridge runtime and `NavDPCommandSource`.
- `src/runtime/planning.py`: shared planner session and trajectory update state.
- `src/runtime/g1_bridge_args.py`: bridge CLI contract.
- `src/runtime/g1_isaac.py`: legacy Isaac runtime compatibility flow.
- `src/locomotion/runtime.py`: locomotion loop and ONNX controller orchestration.
- `src/control/async_planners.py`: async planner workers.
- `src/control/trajectory_tracker.py`: trajectory follower.
- `src/adapters/navdp_http.py`: NavDP HTTP client plus compatibility re-exports for transform helpers.
- `src/adapters/dual_http.py`: dual-system HTTP client.
- `src/adapters/d455_sensor.py`: D455 sensor adapter.
- `src/apps/navdp_server_app.py`: NavDP Flask app.
- `src/apps/dual_server_app.py`: dual-system Flask app.
- `src/services/navdp_inference_service.py`: NavDP inference service.
- `src/services/dual_orchestrator.py`: dual orchestrator service.
