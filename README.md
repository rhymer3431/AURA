# AURA

AURA는 자연어를 이해하고, 본 것을 기억하고, 지금 해야 할 행동을 결정해 실제 움직임으로 연결하는 에이전트 런타임입니다.

이 저장소의 핵심은 "좋은 응답을 생성하는 모델"이 아니라, 로봇이 보고, 기억하고, 판단하고, 움직이는 전 과정을 하나의 실행 구조로 묶는 데 있습니다. 즉 AURA는 단일 모델 저장소가 아니라, 실제 로봇 이식과 Isaac Sim 검증을 함께 고려한 runtime repository입니다.

## What AURA Builds

AURA는 다음과 같은 에이전트 루프를 구현하는 것을 목표로 합니다.

1. 환경을 본다.
2. 관측 결과를 구조화된 메모리에 축적한다.
3. 자연어 지시를 현재 상태와 과거 기억에 연결해 해석한다.
4. 목표를 선택하고 재계획 여부를 판단한다.
5. 최종 행동 명령을 만들고 이동/주시/탐색으로 실행한다.
6. 결과를 다시 상태와 메모리에 반영한다.

예를 들면 다음과 같은 지시를 다룹니다.

- `"아까 봤던 사과를 찾아가"`
- `"따라와"`
- `"보라색 상자를 찾고, 충분히 가까워지면 멈춰"`

중요한 점은 자연어의 출력이 텍스트 응답으로 끝나지 않는다는 것입니다. AURA에서 자연어 이해의 결과는 결국 `target object_id`, `target world pose`, action command, runtime state transition으로 이어져야 합니다.

## Demo

`"보라색 상자를 찾고, 충분히 가까워지면 멈춰"`라고 입력했을 때 실제로 찾아가는 모습

![object find demo](./media/object-find.webp)

## Why This Repo Exists

AURA는 다음 계층을 한 저장소 안에서 연결합니다.

- `perception`
  detector, tracker, depth projection, object mapping, observation fusion
- `memory`
  spatial, temporal, episodic, semantic, working memory
- `services`
  object search, follow, attention, semantic consolidation, memory policy
- `server` / `runtime`
  task ingress, planning coordination, safety, command arbitration, snapshot publication
- `dashboard` / `webrtc` / `ipc`
  운영 제어, 상태 관찰, 스트리밍, transport abstraction
- `training` / `artifacts`
  memory policy 및 System 2 관련 학습 코드와 seed dataset

즉 이 저장소는 "모델 하나를 잘 돌리는 코드"보다, 여러 인지 기능을 행동 가능한 에이전트로 묶는 실행 기반에 가깝습니다.

## Canonical Architecture

노션 위키 기준 현재 AURA의 canonical 구조는 두 문장으로 요약됩니다.

- write-side owner: `MainControlServer`
- read-side truth source: `WorldStateSnapshot`

이 원칙은 왜 중요한가:

- 어떤 task가 활성 상태인지
- 이번 tick에서 retrieve / replan / retry / safe-stop 중 무엇을 할지
- perception, memory, planner 결과를 어떤 상태로 commit할지
- 최종 actuator command를 무엇으로 낼지

이 판단을 여러 모듈이 나눠서 들고 있지 않고, server 계층이 canonical하게 소유해야 자연어 지시 기반 에이전트가 흔들리지 않습니다.

현재 상위 실행 흐름은 다음과 같습니다.

1. 센서 프레임과 task event가 runtime gateway로 들어옵니다.
2. `MainControlServer.tick(...)`가 현재 task와 recovery state를 기준으로 이번 tick의 정책을 정합니다.
3. `PlannerCoordinator`가 perception, memory, planning, nav, locomotion 경로를 조율합니다.
4. 결과는 `WorldStateStore`에 commit됩니다.
5. `CommandResolver`가 최종 action command와 status를 만듭니다.
6. `WorldStateSnapshot`이 dashboard, WebRTC, runtime mirror에 배포됩니다.

이 구조 덕분에 AURA는 "자연어 명령 -> 모델 응답" 수준이 아니라, "자연어 명령 -> 상태 해석 -> 기억 검색 -> 목표 선택 -> 행동 실행 -> 관찰 반영"까지 이어지는 에이전트 루프를 구현할 수 있습니다.

## Memory As Agent Capability

이 저장소에서 메모리는 부가 기능이 아닙니다. 에이전트가 현재 프레임 밖의 세계를 계속 다루기 위한 핵심 계층입니다.

노션 위키의 메모리 정의를 기준으로 AURA의 메모리는 아래 세 가지를 함께 다룹니다.

- 현재 추적 상태
- 장기 객체 기록
- 검색 인덱스

구조적으로는 다음 계층으로 나뉩니다.

- Working Memory
  현재 프레임과 최근 수초의 live state
- Episodic Object Memory
  장기 객체 기록과 canonical object identity
- Semantic Retrieval Index
  자연어 질의와 유사도 검색

그래서 `"아까 봤던 사과를 찾아가"` 같은 지시는 단순 문자열 매칭으로 처리되지 않습니다. 검색 결과는 최종적으로 텍스트가 아니라 `target object_id + target world pose`가 되어 navigation과 action command로 이어져야 합니다.

## Runtime Surfaces

AURA를 읽을 때 기준이 되는 실행 표면은 다음과 같습니다.

- `scripts/run_system.ps1`
  canonical runtime launcher
- `scripts/run_dashboard.ps1`
  운영 및 관찰 UI launcher
- `src/apps/local_stack_app.py`
  단일 프로세스 기반 로컬 스택 진입점
- `src/runtime/aura_runtime.py`
  메인 runtime loop
- `src/server/main_control_server.py`
  write-side canonical control core
- `src/server/world_state_store.py`
  canonical state commit

## Recommended Environment

현재 저장소는 다음 전제를 강하게 가집니다.

- Python `>= 3.10`
- Windows PowerShell 기반 주요 런처 사용
- Isaac Sim / Kit 환경 연동 경로 존재
- detector, nav, system2, dashboard 경로가 한 저장소 안에 함께 존재

실행과 검증은 Windows 11 네이티브 환경을 우선 권장합니다. 특히 Isaac Sim, 센서 경로, PowerShell 런처를 포함한 전체 runtime 검증은 WSL보다 Windows 환경에서 맞춰져 있습니다.

## Quick Start

먼저 editable install 기준의 최소 준비:

```bash
python -m pip install -e .
```

대시보드 백엔드 또는 WebRTC 의존성이 필요하면 선택적으로 extras를 추가합니다.

```bash
python -m pip install -e .[dashboard,webrtc]
```

대표 실행 경로:

```powershell
.\scripts\run_system.ps1
.\scripts\run_system.ps1 -Component runtime
.\scripts\run_dashboard.ps1
```

`run_system.ps1`는 `all`, `nav`, `s2`, `dual`, `runtime` 컴포넌트 모드를 받아 전체 시스템 또는 일부 경로를 올릴 수 있습니다.

## Validation

테스트 구조 자체가 현재 아키텍처 경계를 설명합니다.

- `tests/server`
  state ownership, worker contract, planner coordination, recovery, safety, command resolution 검증
- `tests/services`
  object search, semantic consolidation, memory policy, task orchestration 검증
- `tests/memory`, `tests/perception`
  저장/관측 하위 계층 검증
- `tests/dashboard_backend`, `tests/webrtc`, `tests/ipc`
  observability와 transport surface 검증
- `tests/integration`
  runtime 연결부와 회귀 위험이 큰 흐름 검증
- `tests/training`
  dataset / training 경로 형식 검증

기본 테스트 실행:

```bash
pytest
```

환경 의존성이 큰 경로는 전체 실행보다 관련 범위를 선택적으로 검증하는 편이 현실적입니다.

```bash
pytest tests/server tests/integration
pytest tests/services tests/memory
```

## Directory Guide

- `src/server`
  canonical write-side control core
- `src/runtime`
  runtime gateway, actuator apply, diagnostics
- `src/services`
  자연어 의도 해석 이후의 task policy와 memory-facing service
- `src/memory`
  working, episodic, semantic memory 계층
- `src/perception`
  detector, tracker, depth, observation processing
- `src/inference`
  detector, nav, VLM, training entrypoints
- `src/dashboard_backend`, `src/webrtc`
  운영/관찰 surface
- `src/ipc`
  in-process, ZMQ, shared memory transport
- `scripts`
  Windows 기준 주요 런처
- `docs/architecture`
  canonical architecture 설명
- `tests`
  계층별 검증 코드

## Reading Order

저장소를 빠르게 이해하려면 다음 순서를 권장합니다.

1. 이 `README.md`에서 사용자 관점 목표와 실행 맥락을 확인합니다.
2. [docs/README.md](/mnt/c/Users/mango/project/isaac-aura/docs/README.md)와 [docs/architecture/AURA_RUNTIME_ARCHITECTURE.md](/mnt/c/Users/mango/project/isaac-aura/docs/architecture/AURA_RUNTIME_ARCHITECTURE.md)에서 canonical 구조를 읽습니다.
3. [src/server/main_control_server.py](/mnt/c/Users/mango/project/isaac-aura/src/server/main_control_server.py), [src/server/world_state_store.py](/mnt/c/Users/mango/project/isaac-aura/src/server/world_state_store.py), [src/runtime/aura_runtime.py](/mnt/c/Users/mango/project/isaac-aura/src/runtime/aura_runtime.py) 순으로 실제 runtime 경로를 따라갑니다.
4. `tests/server`, `tests/integration`을 함께 읽어 현재 코드가 무엇을 canonical boundary로 보는지 확인합니다.

## Summary

AURA는 자연어를 이해하는 로봇 에이전트를 만들기 위한 저장소가 아니라, 자연어를 이해한 뒤 실제로 행동하고 상태를 갱신할 수 있는 에이전트 런타임을 만들기 위한 저장소입니다.

핵심은 다음 세 가지입니다.

- 자연어를 행동 가능한 내부 상태로 바꾸는 것
- 현재 관측과 과거 기억을 함께 사용해 목표를 선택하는 것
- 최종 판단을 `MainControlServer`와 `WorldStateSnapshot` 중심의 canonical 구조로 일관되게 운영하는 것
