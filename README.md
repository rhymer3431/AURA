# AURA

AURA는 인간의 인지 파이프라인을 모방한 인지 아키텍처를 구축하고, 이를 실제 휴머노이드 로봇에 이식하는 것을 목표로 하는 프로젝트입니다.

이 프로젝트는 실제 로봇과 센서, 그리고 물리가 구현된 Isaac Sim 시뮬레이터 환경에서 아키텍처를 검증하며, 인식, 구조화 메모리, 자연어 태스크 오케스트레이션, VLM 기반 목표 선택, NavDP 기반 이동 계획을 하나의 저장소 안에서 통합적으로 다룹니다.

주요 목표는 다음과 같습니다.

- 카메라 프레임에서 객체와 사람을 인식하고 추적하기
- 관측 결과를 공간/시간/에피소드/의미 메모리에 축적하기
- `"아까 봤던 사과를 찾아가"`, `"따라와"` 같은 자연어 명령을 행동으로 변환하기
- 인간의 인지 파이프라인을 참고한 인지 아키텍처를 실제 휴머노이드에 이식 가능하도록 설계하기
- 실제 로봇 환경과 Isaac Sim 기반 검증 환경을 함께 지원하기
- 라이브 스모크 진단으로 Isaac 환경 호환성과 센서 파이프라인을 빠르게 점검하기

## Demo

`"보라색 상자를 찾고, 충분히 가까워 지면 멈춰"` 라고 입력했을 때 실제로 찾아가는 모습

![object find demo](./media/object-find.webp)

## 무엇이 들어 있나

이 저장소는 단일 모델 저장소가 아니라, 인지 아키텍처를 실제 로봇과 시뮬레이션 환경에 연결하기 위한 런타임 레포입니다.

포함된 주요 구성요소는 다음과 같습니다.

- `perception`: detector, tracker, depth projection, object mapping, observation fusion
- `memory`: spatial/temporal/episodic/semantic memory와 working memory
- `services`: legacy mission/planning/memory implementations과 compatibility aliases
- `modules`: observation, world model, mission, planning, execution, runtime I/O facade
- `runtime`: `NavigationRuntime`, planning session, supervisor ingress, memory agent runtime
- `apps`: 메모리 에이전트, dashboard backend, WebRTC gateway, deprecated local stack/live smoke shim
- `ipc`: in-process, ZMQ, shared memory transport
- `tests`: 서비스/메모리/인퍼런스/통합 테스트

## 핵심 아키텍처

AURA의 핵심은 하나의 메인 런타임 owner 아래에 인지 파이프라인을 모듈화하는 데 있습니다.

현재 canonical runtime flow는 아래와 같습니다.

1. `run_aura_runtime.ps1`
2. `runtime.navigation_runtime:NavigationRuntime`
3. `ObservationModule.capture()`
4. `WorldModelModule.update()`
5. `MissionModule.update()`
6. `PlanningModule.plan()`
7. `ExecutionModule.execute()`
8. `RuntimeIOModule.publish()`
9. `locomotion.runtime`

이때 ownership은 다음처럼 읽어야 합니다.

- `NavigationRuntime`가 프레임 루프의 유일한 owner다.
- `Supervisor`는 world-model ingress compatibility façade다.
- `MissionManager`/`TaskOrchestrator`는 mission state consumer이며 locomotion controller가 아니다.
- dual-system은 top-level subsystem이 아니라 `PlanningModule` 내부 backend다.
- memory read/write는 `WorldModelModule` 아래 façade로 분리된다.

## 검증 환경

AURA는 실제 로봇에 이식 가능한 구조를 지향하며, 검증은 두 환경에서 수행됩니다.

- 실제 휴머노이드 로봇과 센서를 사용하는 실환경
- 실제 센서 배치와 물리 환경이 반영된 Isaac Sim 시뮬레이터

즉, AURA는 순수 시뮬레이션 전용 프로젝트가 아니라, 실제 로봇 이식을 전제로 시뮬레이터 안에서 반복적으로 검증 가능한 인지 아키텍처 프로젝트입니다.

## 주요 실행 모드

### 1. Navigation Runtime

메인 실행 경로입니다. G1 런타임에서 observation -> world model -> mission -> planning -> execution 순서를 소유합니다.

```powershell
.\scripts\powershell\run_aura_runtime.ps1
```

### 2. Memory Agent

구조화 메모리와 오케스트레이터를 지속 실행하는 에이전트입니다.

```powershell
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --serve
```

### 3. AURA Dashboard / WebRTC

Tauri desktop app으로 AURA runtime stack을 제어하고, WebRTC viewer를 브라우저 대신 데스크톱 셸 안에서 띄웁니다.

```powershell
.\scripts\powershell\run_dashboard.ps1
```

이 런처는 로컬 backend(`127.0.0.1:8095`)와 Tauri dashboard를 함께 실행합니다.

### 4. Experimental Paths

- `system2-memory-lora`
- `memory-policy`
- `text-only memory controller`

이 경로들은 canonical runtime이나 supporting path가 아니라 planner/mission backend 실험 경로로 본다.

### 5. Decommission Targets

- `run_local_stack.ps1` / `apps.local_stack_app`
- `apps.live_smoke_app` / `runtime.live_smoke_runner`

이 surface들은 유지 대상이 아니라 제거 대상이다. 현재는 compatibility shim만 남기고 canonical/supporting 목록에서는 제외했다.

## 환경 전제

- Python `>= 3.10`
- Isaac Sim/Kit 환경을 사용하는 실행 경로가 존재함
- Windows PowerShell 스크립트가 주요 런처 역할을 함
- 일부 경로는 TensorRT, CUDA, D455 asset, llama.cpp/InternVLA 모델 같은 외부 런타임 의존성을 전제로 함

## 권장 실행 환경

이 저장소는 윈도우 11 네이티브 환경에 강하게 의존합니다. 특히 데모 실행과 Isaac 연동 경로를 확인하려면 WSL이나 범용 리눅스 환경보다 Windows 11 위에서 Python 가상환경을 만든 뒤 PowerShell 런처를 통해 실행하는 방식을 권장합니다.

즉, 이 저장소는 순수 Python 라이브러리보다는, 실제 휴머노이드 이식을 목표로 하는 인지 아키텍처 런타임과 검증 도구 모음에 가깝습니다.

## 추천 시작 순서

- 메인 런타임은 `run_aura_runtime.ps1`
- supporting memory path는 `run_memory_agent.ps1`
- 운영/시각화 shell은 `run_dashboard.ps1`
- deprecated/decommission surface인 local stack/live smoke는 새 구조의 시작점으로 사용하지 않는다.

## 디렉터리 가이드

- `src/apps`: 실행 진입점
- `src/runtime`: 런타임 연결부와 시뮬레이터 브리지
- `src/services`: 태스크 로직과 메모리/탐색 서비스
- `src/perception`: 시각 파이프라인
- `src/memory`: 메모리 저장소와 질의 로직
- `src/ipc`: inproc/ZMQ/shared memory 메시징
- `scripts/powershell`: Windows 기준 주요 런처
- `docs/architecture`: 아키텍처 설명 문서
- `tests`: 단위/통합 테스트
- `media`: 데모 자료

## 관련 문서

- [RUNNING.md](./RUNNING.md): 실행 경로와 PowerShell 예시
- [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md): 디렉터리 구조와 책임 분리
- [docs/architecture/RUNTIME_MODES.md](./docs/architecture/RUNTIME_MODES.md): 런타임 모드 요약
- [docs/architecture/DUAL_SYSTEM_PIPELINE.md](./docs/architecture/DUAL_SYSTEM_PIPELINE.md): System 2 -> System 1 계획 흐름
- [docs/architecture/MEMORY_ARCHITECTURE.md](./docs/architecture/MEMORY_ARCHITECTURE.md): 메모리 구조와 업데이트 흐름
- [exts/isaac.aura.live_smoke/docs/README.md](./exts/isaac.aura.live_smoke/docs/README.md): extension 모드 설명

## 테스트

테스트는 서비스, 메모리, IPC, perception, inference, integration 계층으로 나뉘어 있습니다.

```bash
pytest
```

환경 의존성이 큰 경로는 전체 테스트가 아닌 관련 범위만 선택적으로 실행하는 편이 현실적입니다.
