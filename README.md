# Isaac Aura

Isaac Aura는 Isaac Sim 기반 G1 로봇 런타임을 중심으로, 인식, 구조화 메모리, 자연어 태스크 오케스트레이션, VLM 기반 목표 선택, NavDP 기반 이동 계획을 한 저장소에서 묶어 다루는 프로젝트입니다.

주요 목표는 다음과 같습니다.

- 카메라 프레임에서 객체와 사람을 인식하고 추적하기
- 관측 결과를 공간/시간/에피소드/의미 메모리에 축적하기
- `"아까 봤던 사과를 찾아가"`, `"따라와"` 같은 자연어 명령을 행동으로 변환하기
- Isaac Sim 실환경 경로와 synthetic fallback 경로를 모두 지원하기
- 라이브 스모크 진단으로 Isaac 환경 호환성과 센서 파이프라인을 빠르게 점검하기

## Demo

![object find demo](./media/object-find.webp)

애니메이션이 보이지 않으면 직접 열기: [media/object-find.webp](./media/object-find.webp)

## 무엇이 들어 있나

이 저장소는 단일 모델 저장소가 아니라, 아래 구성요소를 함께 포함하는 런타임 레포입니다.

- `perception`: detector, tracker, depth projection, object mapping, observation fusion
- `memory`: spatial/temporal/episodic/semantic memory와 working memory
- `services`: object search, follow, attention, task orchestration, semantic consolidation
- `runtime`: Isaac bridge, planning session, supervisor, live smoke runner
- `apps`: 로컬 스택, 메모리 에이전트, Isaac bridge, live smoke, viewer 실행 진입점
- `ipc`: in-process, ZMQ, shared memory transport
- `tests`: 서비스/메모리/인퍼런스/통합 테스트

## 핵심 아키텍처

기본 실행 흐름은 두 갈래입니다.

1. 인식/메모리 경로  
   센서 프레임이 detector, tracker, depth projection을 거쳐 `MemoryService`와 `TaskOrchestrator`로 들어갑니다.

2. 계획/이동 경로  
   같은 프레임이 System 2(VLM) 목표 선택과 System 1(NavDP) 궤적 생성으로 이어지고, 최종적으로 low-level motion command로 변환됩니다.

태스크 레벨에서는 `TaskOrchestrator`가 다음 행동을 관리합니다.

- 사람 따라가기
- 보이는 객체 접근
- 기억 속 객체 회상 후 탐색
- 발화 방향 주시
- 기본 로컬 탐색

## 주요 실행 모드

### 1. Local Stack

가장 빠르게 전체 파이프라인을 확인하는 경로입니다. 한 프로세스에서 frame source, perception, memory, orchestrator를 함께 실행합니다.

```powershell
.\scripts\powershell\run_local_stack.ps1 --command "아까 봤던 사과를 찾아가"
.\scripts\powershell\run_local_stack.ps1 --command "따라와" --scene person --frame-source auto
```

### 2. Memory Agent

구조화 메모리와 오케스트레이터를 지속 실행하는 에이전트입니다.

```powershell
.\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --serve
```

### 3. Isaac Bridge

Isaac Sim과 연결되는 런타임 진입점입니다. 라이브 부트스트랩이 가능하면 Isaac 경로를 사용하고, 아니면 synthetic fallback으로 내려갑니다.

```powershell
.\scripts\powershell\run_isaac_bridge.ps1 --command "아까 봤던 사과를 찾아가"
```

### 4. G1 Pipeline / Dual System

G1 런타임에서 no-goal roaming, point-goal, 자연어 지시 이후 dual-system 경로를 다룹니다.

```powershell
.\scripts\powershell\run_pipeline.ps1
```

### 5. Live Smoke

Isaac 환경 호환성, D455 센서 마운트, perception ingress, memory ingress를 단계별로 점검하는 진단 경로입니다.

```powershell
.\scripts\powershell\run_live_smoke_preflight.ps1
.\scripts\powershell\run_live_smoke.ps1 --headless
```

## 환경 전제

- Python `>= 3.10`
- Isaac Sim/Kit 환경을 사용하는 실행 경로가 존재함
- Windows PowerShell 스크립트가 주요 런처 역할을 함
- 일부 경로는 TensorRT, CUDA, D455 asset, llama.cpp/InternVLA 모델 같은 외부 런타임 의존성을 전제로 함

즉, 이 저장소는 순수 Python 라이브러리보다는 "Isaac Sim 런타임 + 보조 서비스 + 실험/검증 도구"에 가깝습니다.

## 추천 시작 순서

1. 구조만 빨리 보려면 `run_local_stack.ps1`
2. Isaac 환경 점검이 먼저면 `run_live_smoke_preflight.ps1`
3. 실제 브리지 동작을 보려면 `run_isaac_bridge.ps1`
4. G1 dual-system 흐름까지 보려면 `run_pipeline.ps1`

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
