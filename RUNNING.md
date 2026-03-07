# Running

## Launchers
- G1 bridge: `.\scripts\powershell\run_g1_pointgoal.ps1`
- NavDP sidecar: `.\scripts\powershell\run_navdp_server.ps1`
- Dual orchestrator: `.\scripts\powershell\run_vlm_dual_server.ps1`
- Root compatibility launchers remain available: `.\run_g1_pointgoal.ps1`, `.\run_navdp_server.ps1`, `.\run_vlm_dual_server.ps1`
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
.\run_g1_pointgoal.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0
.\run_g1_pointgoal.ps1 --planner-mode dual --dual-server-url http://127.0.0.1:8890 --instruction "Navigate safely to the target and stop when complete."
```

## Notes
- Canonical PowerShell launchers prepend `PYTHONPATH=<repo>/src` before invoking the new functional modules.
- `play_g1_keyboard_onnx.py` is a compatibility shim that forwards into `locomotion.entrypoint`.
- Background process logs for the dual orchestrator launch flow are written to `tmp/process_logs/`.
