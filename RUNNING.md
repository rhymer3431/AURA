# Running

## Direct Path
- Direct IPC is the default architecture.
- Legacy HTTP remains compatibility-only.

## Local Stack
```powershell
.\scripts\powershell\run_local_stack.ps1 --command "아까 봤던 사과를 찾아가"
.\scripts\powershell\run_local_stack.ps1 --command "따라와" --scene person --frame-source auto
```
- Uses `InprocBus`.
- Runs detector -> memory -> orchestrator -> `ActionCommand` in one process.
- `--frame-source auto` is the default and uses live-first fallback behavior.

## Memory Agent Loopback
```powershell
.\scripts\powershell\run_memory_agent.ps1 --loopback --command "아까 봤던 사과를 찾아가" --frame-source auto
```
- Useful for validating the bridge/agent IPC flow without a second process.

## Two-Process Local Stack
Process 1:
```powershell
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561
```

Process 2:
```powershell
.\scripts\powershell\run_isaac_bridge.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --command "아까 봤던 사과를 찾아가"
```

- Small control messages travel over the ZMQ control plane.
- `FrameHeader` and status telemetry travel over the ZMQ telemetry plane.
- RGB/depth should use `SharedMemoryRing` in 2-process mode.
- If ZMQ or shared memory is unavailable, use the in-process loopback mode.

## Frame Source Modes
```powershell
.\scripts\powershell\run_isaac_bridge.ps1 --frame-source auto
.\scripts\powershell\run_isaac_bridge.ps1 --frame-source live
.\scripts\powershell\run_isaac_bridge.ps1 --frame-source synthetic
```
- `auto`
  - default
  - tries live Isaac input first
  - emits `RuntimeNotice` and falls back to synthetic if live capture is unavailable
- `live`
  - requires live Isaac input
  - startup exits non-zero when live capture is unavailable
- `synthetic`
  - deterministic smoke-test path

## Detector Backend
- Preferred backend order:
  1. `artifacts/models/yoloe-26s-seg-pf.engine`
  2. TensorRT backend if runtime-compatible
  3. color-seg fallback otherwise
- Structured diagnostics are exposed through `DetectorRuntimeReport`.
- In a TensorRT serialization-mismatch environment, the runtime reports the mismatch and falls back cleanly.

## Low-Level G1 Executor
```powershell
.\scripts\powershell\run_g1_pointgoal.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0
.\scripts\powershell\run_g1_pointgoal.ps1 --planner-mode dual --instruction "아까 봤던 사과를 찾아가"
```
- `runtime.g1_bridge` is now a subgoal executor.
- `planning_session.py` uses direct in-process NavDP execution by default.

## Semantic Retrieval Loop
- Task completion creates structured episodes with candidate places, candidate objects, recovery actions, and summary tags.
- `SemanticConsolidationService` converts those episodes into rule-like semantic memory.
- Object recall and follow recovery read those rules back into `WorkingMemory` scoring and `ActionCommand.metadata`.

## Legacy Compatibility
```powershell
.\scripts\powershell\legacy\run_navdp_server.ps1 --port 8888
.\scripts\powershell\legacy\run_vlm_dual_server.ps1 --port 8890 --navdp-url http://127.0.0.1:8888
```

## Current Limits
- TensorRT execution still depends on a matching engine/runtime/CUDA environment.
- Two-process mode is currently single bridge plus single memory agent.
- Live Isaac frame capture outside an initialized Isaac runtime still falls back to synthetic in `auto` mode.
- System2/VLM is optional and not required for the fast path.
