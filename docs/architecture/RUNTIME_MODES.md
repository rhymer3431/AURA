# Runtime Modes

AURA는 하나의 실행 경로만 가진 프로젝트가 아니다. 현재 저장소는 인지 아키텍처를 검증하기 위한 태스크 런타임과, Isaac 환경 적합성을 빠르게 확인하기 위한 진단 런타임을 함께 제공한다.

이 문서는 현재 코드 기준으로 어떤 실행 모드가 존재하는지, 각 모드가 어떤 책임을 가지는지 정리한다.

## 공통 원칙

- 기본 런타임 철학은 direct IPC + structured memory이다.
- 주요 진입점은 `scripts/powershell/*.ps1`이며, Windows 11 네이티브 + PowerShell + Isaac Sim 환경이 기준 경로다.
- `live smoke`는 메인 태스크 실행기가 아니라 Isaac/센서/ingress 진단 경로다.
- 단일 프로세스 디버그와 다중 프로세스 런타임은 같은 메시지/메모리 구조를 최대한 공유한다.

## 1. Local Stack

가장 빠르게 전체 인지 파이프라인을 확인하는 모드다.

- Entry: `scripts/powershell/run_local_stack.ps1`
- Module: `apps.local_stack_app`
- Transport: `InprocBus`
- Frame source: `auto`, `live`, `synthetic`

현재 구조에서 Local Stack은 한 프로세스 안에서 다음을 수행한다.

1. 프레임 소스를 선택한다.
2. `Supervisor`가 perception pipeline을 실행한다.
3. `MemoryService`와 `TaskOrchestrator`가 관측과 명령을 처리한다.
4. 최종 `ActionCommand`를 계산해 로그로 확인한다.

이 모드는 런타임 wiring 확인과 빠른 스모크 테스트에 적합하지만, 브리지/에이전트 분리나 실시간 다중 프로세스 통신을 검증하는 경로는 아니다.

## 2. Memory Agent

구조화 메모리와 태스크 오케스트레이션을 지속 실행하는 에이전트 모드다.

- Entry: `scripts/powershell/run_memory_agent.ps1`
- Module: `apps.memory_agent_app`
- Runtime core: `runtime.memory_agent_runtime`
- Transport: `InprocBus` 또는 `ZmqBus` + `SharedMemoryRing`

Memory Agent의 현재 책임은 다음과 같다.

- `TaskRequest`, `FrameHeader`, `ActionStatus`를 수신한다.
- `Supervisor.run_bus_cycle_result()`를 통해 perception -> memory -> orchestration을 수행한다.
- `ActionCommand`, `HealthPing`, `CapabilityReport`를 발행한다.
- 필요 시 SQLite 메모리 스냅샷을 주기적으로 저장한다.
- 부트스트랩 semantic rule을 로드해 기본 검색 힌트를 제공한다.

주요 실행 패턴은 세 가지다.

- `--serve`: 장시간 메모리 에이전트를 유지
- `--once`: 단발성 사이클 실행
- `--loopback`: 외부 브리지 없이 자체 frame source로 동작

## 3. Internal Frame Bridge

Isaac Sim live frame과 shared-memory/IPC wiring을 붙이는 내부 브리지 경로다.

- Public launcher: 없음
- Internal modules:
  - `apps.frame_bridge_app`
  - `apps.frame_bridge_editor_app`
  - `runtime.frame_bridge_runtime`
  - `runtime.frame_editor_bridge`

이 경로는 editor attach와 내부 런타임 재사용을 위해 남겨 두는 보조 surface다. public 실행 진입점으로 문서화하지 않으며, 런타임 branding도 `isaac bridge`가 아니라 internal frame bridge로 본다.

## 4. AURA Runtime / Dual-System Runtime

G1 런타임은 저수준 이동과 dual-system planning을 직접 구동하는 별도 경로다.

- Entry: `scripts/powershell/run_aura_runtime.ps1`
- Module: `runtime.aura_runtime`
- Planner core: `runtime.planning_session`
- Execution core: `runtime.subgoal_executor`

현재 PowerShell 런처가 노출하는 planner mode는 다음 둘이다.

- `interactive`
- `pointgoal`

코드 레벨에는 `planner-mode=dual`도 존재하지만, `run_aura_runtime.ps1`는 직접 노출하지 않는다. 따라서 일반적인 사용 흐름은 아래와 같다.

- `interactive`: no-goal roaming으로 시작하고, 터미널 자연어 입력 이후 dual-system 경로로 전환
- `pointgoal`: System 2 없이 직접 point-goal planning 수행

또한 launch mode는 다음 개념으로 나뉜다.

- `gui`
- `headless`
- `g1_view`

`g1_view`에서는 동일 프레임이 viewer용 IPC/ZMQ/SHM 경로로도 공개된다.

중요한 점은 G1 pipeline에서 `Supervisor`가 perception/memory 업데이트를 계속 수행하더라도, 실제 저수준 이동 명령은 `PlanningSession`과 `SubgoalExecutor`가 주도한다는 것이다. 즉, 이 경로에서 `TaskOrchestrator`는 병렬 관측 소비자이지 locomotion의 직접 컨트롤러는 아니다.

## 5. Live Smoke

Live Smoke는 Isaac 환경 적합성, D455 센서 마운트, perception ingress, memory ingress를 단계적으로 검증하는 진단 경로다.

- Entry:
  - `scripts/powershell/run_live_smoke_preflight.ps1`
  - `scripts/powershell/run_live_smoke.ps1`
  - `scripts/powershell/run_live_smoke_attach.ps1`
  - `scripts/powershell/run_live_smoke_extension.ps1`
- Modules:
  - `apps.live_smoke_app`
  - `apps.editor_smoke_entry`
  - `runtime.live_smoke_runner`
  - `exts/isaac.aura.live_smoke`

공식 launch mode는 다음과 같다.

- `standalone_python`
- `editor_assisted`
- `extension_mode`

호환성용 deprecated alias도 존재한다.

- `full_app_attach` -> `editor_assisted`

주요 bootstrap profile은 다음과 같다.

- `minimal_headless_sensor_smoke`
- `standalone_render_warmup`
- `full_app_editor_assisted`
- `extension_in_editor`

주요 smoke tier는 다음과 같다.

- `sensor`
- `pipeline`
- `memory`
- `full`

Live Smoke는 detector 품질을 평가하는 문서화된 벤치마크가 아니라, “Isaac 환경에서 프레임이 들어오고 perception/memory ingress가 실제로 연결되어 있는가”를 빠르게 진단하는 경로다.

## 6. Transport와 Frame Source 선택

현재 런타임이 공유하는 전송/소스 원칙은 다음과 같다.

- `InprocBus`
  - 단일 프로세스 디버그와 loopback용
- `ZmqBus`
  - bridge/agent 분리 시 사용
  - control plane과 telemetry plane을 분리
- `SharedMemoryRing`
  - RGB/depth 같은 큰 payload 전달용
- frame source
  - `live`: Isaac 센서 경로 강제
  - `synthetic`: synthetic frame 강제
  - `auto`: live 우선, 불가 시 fallback

즉, 실행 모드는 달라도 observation batch와 message contract는 최대한 같은 형태를 유지하도록 설계되어 있다.

## 현재 한계

- `run_aura_runtime.ps1`는 아직 `planner-mode=dual`을 직접 노출하지 않는다.
- `editor_assisted`와 `extension_mode`는 외부 프로세스 attach가 아니라, 이미 떠 있는 Isaac Full App / Kit 내부 실행을 전제로 한다.
- standalone headless live smoke 성공 여부는 여전히 Isaac/Kit/rendering 환경 일치 여부에 민감하다.
- 일부 경로는 TensorRT, CUDA, Isaac asset, 외부 모델 런타임 의존성을 강하게 가진다.
