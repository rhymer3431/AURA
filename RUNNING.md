# Running

## Launchers
- G1 bridge: `.\scripts\powershell\run_g1_pointgoal.ps1`
- G1 warehouse object-search demo: `.\scripts\powershell\run_g1_object_search_demo.ps1`
- NavDP sidecar: `.\scripts\powershell\run_navdp_server.ps1`
- Dual orchestrator: `.\scripts\powershell\run_vlm_dual_server.ps1`
- Root compatibility launchers remain available: `.\run_g1_pointgoal.ps1`, `.\run_g1_object_search_demo.ps1`, `.\run_navdp_server.ps1`, `.\run_vlm_dual_server.ps1`
- Python compatibility bridge entrypoint remains `python -m navdp.g1_bridge`

## Defaults
- Canonical bridge module: `runtime.g1_bridge`
- Canonical NavDP server module: `apps.navdp_server_app`
- Canonical dual server module: `apps.dual_server_app`
- Default G1 USD: `src/locomotion/g1/g1_d455.usd`
- Compatibility fallback G1 USD: `g1_play/g1/g1_d455.usd`

## Example Commands
```powershell
.\run_navdp_server.ps1 --port 8888 --checkpoint .\artifacts\models\navdp-weights.ckpt
.\run_vlm_dual_server.ps1 --port 8890 --navdp-url http://127.0.0.1:8888 --vlm-url http://127.0.0.1:8080 --s2-mode auto
.\run_g1_pointgoal.ps1 --planner-mode interactive
.\run_g1_pointgoal.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0
.\run_g1_pointgoal.ps1 --planner-mode dual --dual-server-url http://127.0.0.1:8890 --instruction "Navigate safely to the target and stop when complete."
.\run_g1_object_search_demo.ps1
.\run_g1_object_search_demo.ps1 --demo-object-x 3.0 --demo-object-y -1.0 --object-stop-radius-m 0.9
```

## Notes
- Canonical PowerShell launchers prepend `PYTHONPATH=<repo>/src` before invoking the new functional modules.
- `run_g1_pointgoal.ps1` now defaults to `--planner-mode interactive`, which keeps roaming with `nogoal_step` until you type a natural-language command into the same terminal.
- Interactive terminal commands: `/help`, `/cancel`, `/quit`.
- The object-search demo spawns a bright red cube at `(2.0, 0.0)` by default, reuses `planner-mode=dual`, and exits on `dual stop` or when the robot enters the default `0.8m` stop radius.
- Object-search override flags: `--demo-object-x`, `--demo-object-y`, `--demo-object-size-m`, `--object-stop-radius-m`, and `--instruction`.
- Expected object-search logs include a placement line like `[G1_OBJECT_SEARCH] demo object placed ...` and step lines like `[G1_OBJECT_SEARCH][step=...] object_dist=... goal_v=... traj_v=...`.
- Interactive logs use `[G1_INTERACTIVE][ROAM]` while wandering and `[G1_INTERACTIVE][TASK]` while executing the latest natural-language command.
- `play_g1_keyboard_onnx.py` is a compatibility shim that forwards into `locomotion.entrypoint`.
- Background process logs for the dual orchestrator launch flow are written to `tmp/process_logs/`.
