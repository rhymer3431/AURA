# NavDP G1 PointGoal

## Entry Surfaces
- Canonical launcher: `.\scripts\powershell\run_aura_runtime.ps1`
- Canonical object-search demo launcher: `.\scripts\powershell\run_g1_object_search_demo.ps1`
- Deprecated compatibility launcher: `.\run_g1_pointgoal.ps1`
- Compatibility object-search demo launcher: `.\run_g1_object_search_demo.ps1`
- Compatibility ONNX play shim: `play_g1_keyboard_onnx.py`

## Runtime Flow
- Runtime flow: `runtime.navigation_runtime -> modules.{observation,world_model,mission,planning,execution,runtime_io} -> locomotion.runtime`
- Canonical robot asset path: `src/locomotion/g1/g1_d455.usd`
- Compatibility fallback robot asset path: `g1_play/g1/g1_d455.usd`

## Common Commands
```powershell
.\run_navdp_server.ps1 --port 8888 --checkpoint .\artifacts\models\navdp-weights.ckpt
.\run_vlm_dual_server.ps1 --port 8890 --navdp-url http://127.0.0.1:8888 --vlm-url http://127.0.0.1:8080 --s2-mode auto
.\scripts\powershell\run_aura_runtime.ps1 --planner-mode interactive --launch-mode gui
.\scripts\powershell\run_aura_runtime.ps1 --scene-preset warehouse --planner-mode interactive --launch-mode gui
.\scripts\powershell\run_aura_runtime.ps1 --scene-preset interioragent --planner-mode interactive --launch-mode gui
.\scripts\powershell\run_aura_runtime.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0 --server-url http://127.0.0.1:8888
.\scripts\powershell\run_aura_runtime.ps1 --planner-mode pointgoal --launch-mode g1_view --goal-x 2.0 --goal-y 0.0 --show-depth
.\scripts\powershell\run_aura_runtime.ps1 --scene-preset "interior agent kujiale 3" --planner-mode pointgoal --goal-x 3.5 --goal-y -1.0 --global-map-image ".\datasets\InteriorAgent\kujiale_0003\occupancy map.png"
.\scripts\powershell\run_aura_runtime.ps1 --scene-preset "interior agent kujiale 3" --planner-mode pointgoal --goal-x 3.5 --goal-y -1.0 --global-map-image ".\datasets\InteriorAgent\kujiale_0003\occupancy map.png" --global-map-config ".\datasets\InteriorAgent\kujiale_0003\config.txt" --global-waypoint-spacing-m 0.75 --global-inflation-radius-m 0.25
```

## Notes
- The canonical runtime class is `runtime.navigation_runtime:NavigationRuntime`.
- `runtime.aura_runtime` remains as a deprecated compatibility wrapper.
- Locomotion code and G1 assets are canonical under `src/locomotion`.
- `TaskOrchestrator` now reads as a mission-module compatibility alias.
- `DualOrchestrator` now reads as a planning-coordinator compatibility alias.
- `planner-mode=interactive` starts in no-goal roaming and accepts natural-language commands from the same terminal.
- `launch-mode=gui` keeps the existing Isaac GUI runtime path.
- `launch-mode=g1_view` runs Isaac headless, publishes D455 frames through the production ZMQ/shared-memory IPC path, and auto-starts the external OpenCV viewer with YOLO overlays.
- `planner-mode=dual` is no longer exposed by the pipeline launcher.
- `--scene-preset warehouse` keeps the Isaac Simple Warehouse default scene.
- `--scene-preset interioragent` loads `datasets\InteriorAgent\kujiale_0004\kujiale_0004_navila_sanitized.usda`.
- `--global-map-image` enables an opt-in A* global route layer for `planner-mode=pointgoal`; if `--global-map-config` is omitted, the runtime resolves `config.txt` next to the map image.
- Interactive slash commands: `/help`, `/cancel`, `/quit`.
- Interactive mode logs `[G1_INTERACTIVE][ROAM]` during roaming and `[G1_INTERACTIVE][TASK]` while executing the latest terminal instruction.
