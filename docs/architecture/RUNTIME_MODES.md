# Runtime Modes

AURA의 새 기준 구조는 하나의 메인 런타임 owner와 그 주변 supporting/experimental/decommission 경로를 분리하는 것이다.

## Canonical Runtime

- Entry: `scripts/powershell/run_aura_runtime.ps1`
- Module: `runtime.navigation_runtime`
- Runtime owner: `NavigationRuntime`

Canonical flow는 아래 순서를 따른다.

1. `ObservationModule.capture()`
2. `WorldModelModule.update()`
3. `MissionModule.update()`
4. `PlanningModule.plan()`
5. `ExecutionModule.execute()`
6. `RuntimeIOModule.publish()`
7. `locomotion.runtime`

핵심 ownership 원칙은 다음과 같다.

- `NavigationRuntime`가 프레임 루프의 유일한 owner다.
- `Supervisor`는 world-model ingress compatibility façade다.
- `MissionManager`/`TaskOrchestrator`는 mission state consumer다.
- dual-system은 top-level architecture가 아니라 planning backend다.
- memory read/write는 world model 아래 façade로 분리한다.

## Supporting Paths

다음 경로는 canonical runtime을 보조하지만 메인 주행 경로는 아니다.

- `scripts/powershell/run_memory_agent.ps1`
- `apps.memory_agent_app`
- `runtime.memory_agent_runtime`
- `scripts/powershell/run_dashboard.ps1`
- `apps.dashboard_backend_app`
- `apps.webrtc_gateway_app`
- `dashboard/`
- `dashboard_backend/`
- `webrtc/`

정리 기준은 다음과 같다.

- memory agent는 loopback/IPC 기반 supporting runtime이다.
- dashboard/WebRTC는 canonical runtime을 제어하고 관찰하는 shell이다.
- internal frame bridge는 public surface가 아니라 내부 supporting seam이다.

## Experimental Paths

다음 경로는 canonical/supporting이 아니라 실험 경로로 취급한다.

- `system2-memory-lora`
- `memory-policy`
- `text-only memory controller`
- training/dataset tooling under `src/inference/training`

이들은 planning 또는 mission backend experiment로만 소개해야 하며 메인 런타임 surface로 노출하지 않는다.

## Decommission Targets

다음 경로는 유지 대상이 아니라 제거 대상이다.

- `scripts/powershell/run_local_stack.ps1`
- `apps.local_stack_app`
- `apps.live_smoke_app`
- `apps.editor_smoke_entry`
- `runtime.live_smoke_runner`
- live-smoke extension/doc/test surface

현재 단계에서의 원칙은 다음과 같다.

- public import/launcher 호환성은 thin shim으로 유지한다.
- 문서상 canonical/supporting 목록에서는 제외한다.
- 필요한 최소 기능은 `NavigationRuntime` 또는 `MemoryAgentRuntime --loopback` 같은 대체 경로로 흡수한다.

## 현재 한계

- deprecated live-smoke 문서와 테스트는 아직 후속 정리가 더 필요하다.
- `TaskOrchestrator`와 `DualOrchestrator`는 이름상 legacy path에 남아 있지만, 새 ownership은 각각 mission/planning으로 읽어야 한다.
- `control` 패키지에는 mission/planning/execution helper가 아직 혼재해 있으므로 물리 이동은 후속 phase에서 진행한다.
