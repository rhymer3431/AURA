# Running

## Canonical Runtime

```powershell
.\scripts\powershell\run_aura_runtime.ps1
.\scripts\powershell\run_aura_runtime.ps1 --planner-mode interactive --launch-mode gui
.\scripts\powershell\run_aura_runtime.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0
```

- `run_aura_runtime.ps1` now targets `runtime.navigation_runtime`.
- `NavigationRuntime` is the single owner of the frame loop.
- dual-system은 별도 top-level mode가 아니라 planning backend로 유지된다.

## Supporting Paths

### Memory Agent

```powershell
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --serve --agent-id memory_agent_a
.\scripts\powershell\run_memory_agent.ps1 --loopback --frame-source auto --once
```

- structured memory와 mission loop를 supporting path로 유지한다.
- 빠른 loopback 진단은 deprecated local stack 대신 이 경로를 우선 사용한다.

### Dashboard / Viewer

```powershell
.\scripts\powershell\run_dashboard.ps1
```

- dashboard backend와 WebRTC viewer는 canonical runtime을 제어/관찰하는 supporting shell이다.

## Experimental Paths

- `system2-memory-lora`
- `memory-policy`
- `text-only memory controller`

이들은 canonical runtime 실행 순서가 아니라 planner/mission backend 실험 경로다.

## Deprecated / Decommission

### Local Stack

```powershell
.\scripts\powershell\run_local_stack.ps1 --command "아까 봤던 사과를 찾아가"
```

- deprecated compatibility shim만 남아 있다.
- canonical/supporting 목록에서는 제외한다.

### Live Smoke

- live-smoke app/runtime은 decommission 대상이다.
- 현재 저장소의 PowerShell launcher inventory에는 `run_live_smoke*.ps1`가 없다.
- 관련 코드는 compatibility/decommission 정리 대상이며 메인 실행 surface로 사용하지 않는다.

## Detector Backend

- preferred detector path는 base detection model이다.
- TensorRT/CUDA/Isaac asset 의존성은 여전히 환경에 민감하다.

## Current Limits

- deprecated live-smoke 문서/테스트는 후속 정리가 더 필요하다.
- `control` 패키지에는 mission/planning/execution helper가 아직 함께 남아 있다.
- multi-agent command arbitration은 아직 targeted routing이 아니다.
