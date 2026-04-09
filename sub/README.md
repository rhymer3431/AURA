# Subsystem Architecture

This runtime is divided into five subsystem views under `src/sub` and documented here under `/sub`.

1. `Navigation Subsystem`
   - Owns NavDP planning/execution, point-goal control, trajectory following, and NavDP HTTP serving.
   - Primary source modules: `src/g1_play/navdp_runtime.py`, `src/g1_play/navdp_client.py`, `src/g1_play/navdp_follower.py`, `src/navdp/navdp_server.py`.

2. `Inference Subsystem`
   - Owns InternVLA / llama.cpp inference serving and multimodal grounding helpers.
   - Primary source modules: `serve_internvla_nav_server.py`, `src/g1_play/internvla_nav.py`, `src/g1_play/tasking/llm_client.py`.

3. `World State Subsystem`
   - Owns camera sensing, runtime state snapshots, scene/asset resolution, and camera pitch APIs.
   - Primary source modules: `src/g1_play/camera_control/*`, `src/g1_play/camera_api.py`, `src/g1_play/navdp_runtime.py`, `src/g1_play/paths.py`.

4. `Planner Subsystem`
   - Owns task-frame normalization, planner endpoint calls, subgoal expansion, validation, and reporting.
   - Primary source modules: `src/g1_play/tasking/*`, `scripts/serve_planner_qwen3_nothink.ps1`.

5. `Control Subsystem`
   - Owns runtime orchestration, command ingestion, locomotion control, and launch surfaces.
   - Primary source modules: `src/g1_play/runtime.py`, `src/g1_play/controller.py`, `src/g1_play/command.py`, `play_g1_internvla_navdp.py`, `run_sim_g1_internvla_navdp_windows.bat`.

The code-facing manifests live in `src/sub/<subsystem>/__init__.py`. They intentionally use lazy imports so the architecture map remains importable without forcing Isaac Sim or heavyweight runtime dependencies during basic tooling and tests.
