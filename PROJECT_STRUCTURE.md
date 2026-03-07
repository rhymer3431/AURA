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
├── llama.cpp/
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
├── tests/
├── tmp/
│   └── process_logs/
├── agents.md
├── INTERNVLA_SYSTEM2_SETUP.md
├── MIGRATION_LOG.md
├── NAVDP_G1_POINTGOAL.md
├── pyproject.toml
├── REFACTOR_NOTES.md
├── PROJECT_STRUCTURE.md
└── RUNNING.md
```

## Folder Responsibilities
- `artifacts/models`: runtime model artifacts such as ONNX, checkpoint, and GGUF weights.
- `docs/refactor_baseline`: refactor reference snapshots, default args, and migration support files.
- `experiments/launchers`: experiment-specific launcher scripts that are not part of the default runtime path.
- `llama.cpp`: bundled llama.cpp Windows binaries and shared libraries used by the VLM/System2 flows.
- `logs`: runtime log output directory used by launcher flows.
- `scripts/powershell`: canonical PowerShell launchers for the supported runtime modes.
- `src/adapters`: external system boundaries including HTTP clients and D455 sensor integration.
- `src/apps`: Flask app entry modules for the NavDP server and dual-system server.
- `src/common`: shared geometry and scene helpers used across runtime layers.
- `src/control`: planner coordination and trajectory tracking logic.
- `src/inference`: policy-network loading and inference-only model code.
- `src/locomotion`: ONNX locomotion runtime, controller, command handling, and G1-specific assets/config.
- `src/runtime`: high-level bridge runtime, CLI contract, Isaac compatibility flow, and planning state.
- `src/services`: request orchestration and service-layer business logic.
- `src/vendor`: bundled third-party code and licenses isolated from first-party modules.
- `tests`: focused unit tests for orchestration, bridge args, object-search helpers, and tracking logic.
- `tmp/process_logs`: launcher-generated stdout/stderr captures for local process orchestration.

## Entrypoints
- `run_g1_pointgoal.ps1`: root compatibility launcher that delegates to `scripts/powershell/run_g1_pointgoal.ps1`.
- `run_g1_object_search_demo.ps1`: root compatibility launcher that delegates to `scripts/powershell/run_g1_object_search_demo.ps1`.
- `run_navdp_server.ps1`: root compatibility launcher that delegates to `scripts/powershell/run_navdp_server.ps1`.
- `run_vlm_dual_server.ps1`: root compatibility launcher that delegates to `scripts/powershell/run_vlm_dual_server.ps1`.
- `run_internvla_system2.ps1`: root compatibility launcher that delegates to `scripts/powershell/run_internvla_system2.ps1`.
- `scripts/powershell/run_g1_pointgoal.ps1`: canonical G1 point-goal bridge launcher.
- `scripts/powershell/run_g1_object_search_demo.ps1`: canonical warehouse object-search demo launcher.
- `scripts/powershell/run_navdp_server.ps1`: canonical NavDP Flask server launcher for `src/apps/navdp_server_app.py`.
- `scripts/powershell/run_vlm_dual_server.ps1`: canonical dual-server launcher for `src/apps/dual_server_app.py`.
- `scripts/powershell/run_internvla_system2.ps1`: canonical InternVLA System2 launcher.
- `python -m locomotion`: package entrypoint backed by `src/locomotion/__main__.py`.

## Key Runtime Modules
- `src/runtime/g1_bridge.py`: bridge runtime, command-source wiring, and high-level point-goal execution flow.
- `src/runtime/g1_bridge_args.py`: bridge CLI argument schema and parsing helpers.
- `src/runtime/g1_isaac.py`: Isaac-side runtime compatibility flow.
- `src/runtime/planning.py`: planner session state and trajectory update coordination.
- `src/locomotion/entrypoint.py`: locomotion runtime entrypoint used by package/module execution.
- `src/locomotion/runtime.py`: locomotion loop and ONNX controller orchestration.
- `src/locomotion/controller.py`: low-level controller behavior for locomotion execution.
- `src/control/async_planners.py`: asynchronous planner worker coordination.
- `src/control/trajectory_tracker.py`: trajectory follower and tracking helpers.
- `src/adapters/navdp_http.py`: NavDP HTTP client and adapter-facing transform helpers.
- `src/adapters/dual_http.py`: dual-system HTTP client for VLM/NavDP integration.
- `src/adapters/d455_sensor.py`: RealSense D455 sensor adapter.
- `src/apps/navdp_server_app.py`: NavDP Flask app entry module.
- `src/apps/dual_server_app.py`: dual-system Flask app entry module.
- `src/services/navdp_inference_service.py`: NavDP inference request handling.
- `src/services/dual_orchestrator.py`: dual-system orchestration service.
