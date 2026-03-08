# NavDP G1 PointGoal

## Entry Surfaces
- Canonical launcher: `.\scripts\powershell\run_pipeline.ps1`
- Canonical object-search demo launcher: `.\scripts\powershell\run_g1_object_search_demo.ps1`
- Compatibility launcher: `.\run_pipeline.ps1`
- Deprecated compatibility launcher: `.\run_g1_pointgoal.ps1`
- Compatibility object-search demo launcher: `.\run_g1_object_search_demo.ps1`
- Compatibility Python entrypoint: `python -m navdp.g1_bridge`
- Compatibility ONNX play shim: `play_g1_keyboard_onnx.py`

## Runtime Flow
- Compatibility flow: `navdp.g1_bridge facade -> runtime.g1_bridge -> locomotion.runtime -> runtime.planning -> control/adapters`
- Canonical robot asset path: `src/locomotion/g1/g1_d455.usd`
- Compatibility fallback robot asset path: `g1_play/g1/g1_d455.usd`

## Common Commands
```powershell
.\run_navdp_server.ps1 --port 8888 --checkpoint .\artifacts\models\navdp-weights.ckpt
.\run_vlm_dual_server.ps1 --port 8890 --navdp-url http://127.0.0.1:8888 --vlm-url http://127.0.0.1:8080 --s2-mode auto
.\run_pipeline.ps1 --planner-mode interactive --launch-mode gui
.\run_pipeline.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0 --server-url http://127.0.0.1:8888
.\run_pipeline.ps1 --planner-mode pointgoal --launch-mode g1_view --goal-x 2.0 --goal-y 0.0 --show-depth
```

## Notes
- The bridge runtime is now canonical under `src/runtime`.
- Locomotion code and G1 assets are canonical under `src/locomotion`.
- Planner, adapter, and tracking logic live directly under the functional packages in `src/`.
- `planner-mode=interactive` starts in no-goal roaming and accepts natural-language commands from the same terminal.
- `launch-mode=gui` keeps the existing Isaac GUI runtime path.
- `launch-mode=g1_view` runs Isaac headless, publishes D455 frames through the production ZMQ/shared-memory IPC path, and auto-starts the external OpenCV viewer with YOLO overlays.
- `planner-mode=dual` is no longer exposed by the pipeline launcher.
- Interactive slash commands: `/help`, `/cancel`, `/quit`.
- Interactive mode logs `[G1_INTERACTIVE][ROAM]` during roaming and `[G1_INTERACTIVE][TASK]` while executing the latest terminal instruction.
