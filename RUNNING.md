# Running

## Default Direct Stack
- Local debug stack: `.\scripts\powershell\run_local_stack.ps1`
- Memory agent only: `.\scripts\powershell\run_memory_agent.ps1`
- Isaac bridge scaffold: `.\scripts\powershell\run_isaac_bridge.ps1`
- Optional System2 helper: `.\scripts\powershell\run_system2_optional.ps1`

## Low-Level Executor
- G1 bridge / pointgoal executor: `.\scripts\powershell\run_g1_pointgoal.ps1`
- Object-search demo compatibility flow: `.\scripts\powershell\run_g1_object_search_demo.ps1`

## Legacy HTTP Compatibility
- NavDP Flask sidecar: `.\scripts\powershell\legacy\run_navdp_server.ps1`
- Dual Flask orchestrator: `.\scripts\powershell\legacy\run_vlm_dual_server.ps1`
- Compatibility wrappers remain available at:
  - `.\scripts\powershell\run_navdp_server.ps1`
  - `.\scripts\powershell\run_vlm_dual_server.ps1`
  - root wrappers `.\run_navdp_server.ps1` and `.\run_vlm_dual_server.ps1`

## Example Commands
```powershell
.\scripts\powershell\run_local_stack.ps1 --command "아까 봤던 사과를 찾아가"
.\scripts\powershell\run_memory_agent.ps1 --memory-db-path .\state\memory\memory.sqlite
.\scripts\powershell\run_isaac_bridge.ps1 --command "따라와"
.\scripts\powershell\run_g1_pointgoal.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0
.\scripts\powershell\legacy\run_navdp_server.ps1 --port 8888 --checkpoint .\artifacts\models\navdp-weights.ckpt
.\scripts\powershell\legacy\run_vlm_dual_server.ps1 --port 8890 --navdp-url http://127.0.0.1:8888
```

## Notes
- `PYTHONPATH=<repo>\src` is still the expected import root for direct Python execution.
- The direct stack currently uses `InprocBus` by default and can be upgraded to `ZmqBus`/shared memory without changing the task-orchestrator interface.
- The memory database path defaults to `state/memory/memory.sqlite`.
- Legacy HTTP launchers are preserved for compatibility, but the direct memory/task path is now the primary architecture.
