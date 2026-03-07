# Running

## Direct Path
- Direct IPC is the default architecture.
- Legacy HTTP remains compatibility-only.

## Local Stack
```powershell
.\scripts\powershell\run_local_stack.ps1 --command "아까 봤던 사과를 찾아가"
.\scripts\powershell\run_local_stack.ps1 --command "따라와" --scene person
```
- Uses `InprocBus`.
- Runs detector -> memory -> orchestrator -> `ActionCommand` in one process.

## Memory Agent Loopback
```powershell
.\scripts\powershell\run_memory_agent.ps1 --loopback --command "아까 봤던 사과를 찾아가"
```
- Useful for validating the bridge/agent IPC flow without a second process.

## Two-Process Local Stack
Process 1:
```powershell
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --bind --endpoint tcp://127.0.0.1:5560
```

Process 2:
```powershell
.\scripts\powershell\run_isaac_bridge.ps1 --bus zmq --connect --endpoint tcp://127.0.0.1:5560 --command "아까 봤던 사과를 찾아가"
```

- Small messages travel over `ZmqBus`.
- RGB/depth should use `SharedMemoryRing` in 2-process mode.
- If ZMQ or shared memory is unavailable, use the in-process loopback mode.

## Low-Level G1 Executor
```powershell
.\scripts\powershell\run_g1_pointgoal.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0
.\scripts\powershell\run_g1_pointgoal.ps1 --planner-mode dual --instruction "아까 봤던 사과를 찾아가"
```
- `runtime.g1_bridge` is now a subgoal executor.
- `planning_session.py` uses direct in-process NavDP execution by default.

## Detector Backend
- Preferred backend order:
  1. `artifacts/models/yoloe-26s-seg-pf.engine`
  2. TensorRT backend if runtime-compatible
  3. color-seg fallback otherwise
- Fallback remains the expected path in environments without TensorRT or with incompatible engines.

## Legacy Compatibility
```powershell
.\scripts\powershell\legacy\run_navdp_server.ps1 --port 8888
.\scripts\powershell\legacy\run_vlm_dual_server.ps1 --port 8890 --navdp-url http://127.0.0.1:8888
```

## Current Limits
- TensorRT YOLOE decode/post-processing is still TODO, so engine load may warn and fall back.
- Two-process mode is currently single bridge plus single memory agent.
- System2/VLM is optional and not required for the fast path.
