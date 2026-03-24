# AURA

AURA는 로봇이 자연어 지시를 이해하고 실제 행동으로 수행하도록 만드는 런타임 프로젝트다. 이 저장소는 단일 모델을 서빙하는 저장소가 아니라, 인식, 메모리, 계획, 이동, 관찰을 하나의 실행 구조로 묶는 데 초점을 둔다.

핵심 목표는 세 가지다.

- 자연어 지시를 행동 가능한 내부 상태와 명령으로 변환하기
- 현재 관측과 과거 기억을 함께 사용해 목표를 선택하고 재계획하기
- 실제 로봇과 Isaac Sim 환경에서 같은 런타임 구조를 검증하기

## 데모

`"보라색 상자를 찾고, 충분히 가까워지면 멈춰"`라고 입력했을 때 실제로 찾아가는 모습

![object find demo](./media/object-find.webp)

## 프로젝트 목적

AURA가 다루는 문제는 "모델이 무엇을 말할 것인가"보다 "로봇이 다음에 무엇을 할 것인가"에 가깝다.

이 프로젝트는 다음 질문에 답하는 런타임을 만드는 것을 목표로 한다.

- 로봇이 지금 무엇을 보고 있는가
- 과거에 본 객체와 사람을 어떻게 기억할 것인가
- 사용자의 자연어 지시를 현재 상태와 어떻게 연결할 것인가
- 계획 실패, 관측 누락, 안전 이벤트가 있을 때 어떻게 재판단할 것인가
- 최종 actuator command를 어떤 기준으로 확정할 것인가

즉 AURA는 자연어 인터페이스를 가진 로봇 에이전트를 위한 실행 기반이며, perception, memory, planning, control, observability를 하나의 구조로 통합한다.

## 핵심 아키텍처 원칙

현재 AURA의 구조는 두 원칙으로 요약된다.

- 모든 write-side 판단과 상태 전이는 `MainControlServer`가 소유한다
- 모든 read-side 관찰은 `WorldStateSnapshot`을 기준으로 노출한다

이 원칙은 task lifecycle, recovery state, planner 결과, safety 상태, 최종 command가 서로 다른 모듈에 흩어져 일관성을 잃는 문제를 막기 위한 것이다. runtime local state나 legacy helper는 더 이상 truth source가 아니다.

상위 구조는 다음과 같다.

```text
Robot / Simulator
    |
    v
AuraRuntimeCommandSource
    |
    | capture frame/task/runtime inputs
    v
MainControlServer.tick(...)
    |
    +-- TaskManager
    +-- DecisionEngine
    +-- PlannerCoordinator
    +-- SafetySupervisor
    +-- CommandResolver
    +-- WorldStateStore
    |
    v
Resolved command + WorldStateSnapshot
    |
    +-- actuator apply
    +-- telemetry publish
    +-- dashboard/webrtc/runtime mirror
```

## 실행 흐름

1. runtime gateway가 센서 프레임, task event, runtime status를 수집한다.
2. `TaskManager`가 현재 planner mode와 task lifecycle을 정리한다.
3. `DecisionEngine`가 retrieve, replan, retry, backoff, System 2 재질의 여부를 계산한다.
4. `PlannerCoordinator`가 perception, memory, planning, nav, locomotion 경로를 조율한다.
5. `SafetySupervisor`가 sensor missing, stale, timeout, safe-stop 같은 안전 이벤트를 반영한다.
6. `WorldStateStore`가 canonical state를 commit한다.
7. `CommandResolver`가 최종 action command와 status를 만든다.
8. `WorldStateSnapshot`이 dashboard, WebRTC, runtime mirror에 배포된다.

이 흐름의 목적은 자연어 지시를 단순 응답으로 끝내지 않고, 상태 해석, 메모리 검색, 목표 선택, 이동 실행, 결과 반영까지 이어지는 폐루프로 만드는 것이다.

## 주요 컴포넌트

- `src/runtime`
  frame capture, task/control ingress, actuator apply, telemetry publish를 담당하는 gateway 계층
- `src/server`
  canonical write-side control core. `MainControlServer`, `TaskManager`, `DecisionEngine`, `PlannerCoordinator`, `SafetySupervisor`, `CommandResolver`, `WorldStateStore`가 위치한다
- `src/perception`
  detector, tracker, depth projection, object mapping, observation fusion으로 관측을 구성한다
- `src/memory`
  working, temporal, spatial, episodic, semantic memory를 통해 현재 상태와 장기 기억을 유지한다
- `src/services`
  object search, follow, attention, semantic consolidation, memory policy 같은 task-level service를 제공한다
- `src/inference`
  detector, nav, VLM, training entrypoint를 포함한다
- `src/dashboard_backend`, `src/webrtc`, `src/ipc`
  운영 제어, 상태 관찰, 스트리밍, transport abstraction을 담당한다

## 메모리와 자연어 태스크

AURA에서 메모리는 부가 기능이 아니라 에이전트 능력의 일부다. 로봇이 현재 프레임 밖의 세계를 계속 다루려면 현재 추적 상태와 장기 객체 기록, 검색 가능한 표현이 함께 필요하다.

메모리 구조는 크게 세 계층으로 나뉜다.

- Working Memory
  현재 프레임과 최근 수초의 live state를 다룬다
- Episodic Object Memory
  객체의 장기 기록과 canonical object identity를 유지한다
- Semantic Retrieval
  자연어 질의와 유사도 검색을 통해 후보를 찾는다

그래서 `"아까 봤던 사과를 찾아가"` 같은 지시는 문자열 매칭으로 처리되지 않는다. 자연어 이해의 결과는 결국 `target object_id`, `target world pose`, action command로 이어져 navigation과 locomotion에 연결되어야 한다.

## 저장소가 담는 범위

AURA는 다음 기능을 한 저장소 안에서 통합한다.

- detector, tracker, depth 기반의 perception
- spatial, temporal, episodic, semantic memory
- 자연어 지시 기반 task orchestration과 planning coordination
- Nav 경로와 locomotion proposal 생성
- runtime safety, recovery, command arbitration
- dashboard, WebRTC, telemetry 기반 observability
- memory policy 및 System 2 관련 학습 코드와 데이터셋

즉 이 저장소는 순수 시뮬레이션 프로젝트도, 순수 모델 저장소도 아니다. 실제 로봇 이식을 염두에 둔 에이전트 런타임과 검증 도구 모음에 가깝다.

## 실행 환경

현재 저장소는 다음 전제를 가진다.

- Python `>= 3.10`
- Windows PowerShell 기반 주요 런처 사용
- Isaac Sim / Kit 환경 연동 경로 존재
- detector, nav, system2, dashboard 경로가 한 저장소 안에 함께 존재

전체 검증은 Windows 11 네이티브 환경을 우선 권장한다. 특히 Isaac Sim, 센서 경로, PowerShell 런처를 포함한 실행 흐름은 WSL보다 Windows 환경에 맞춰져 있다.

## 실행

기본 설치:

```bash
python -m pip install -e .
```

대시보드 또는 WebRTC 의존성이 필요하면 extras를 추가한다.

```bash
python -m pip install -e .[dashboard,webrtc]
```

대표 실행 경로:

```powershell
.\scripts\run_system.ps1
.\scripts\run_system.ps1 -Component runtime
.\scripts\run_dashboard.ps1
```

`run_system.ps1`는 `all`, `nav`, `s2`, `dual`, `runtime` 모드를 받아 전체 시스템 또는 일부 경로를 실행할 수 있다.

## 검증

테스트 구조는 현재 아키텍처 경계를 반영한다.

- `tests/server`
  state ownership, worker contract, planner coordination, recovery, safety, command resolution 검증
- `tests/services`
  object search, memory policy, semantic consolidation, task-level behavior 검증
- `tests/memory`, `tests/perception`
  저장 및 관측 하위 계층 검증
- `tests/dashboard_backend`, `tests/webrtc`, `tests/ipc`
  observability와 transport surface 검증
- `tests/integration`
  runtime 연결부와 회귀 위험이 큰 흐름 검증
- `tests/training`
  dataset 및 training 경로 형식 검증

기본 실행:

```bash
pytest
```

환경 의존성이 큰 경로는 범위를 줄여 검증하는 편이 현실적이다.

```bash
pytest tests/server tests/integration
pytest tests/services tests/memory
```

## 디렉터리 가이드

- `src/server`
  canonical write-side control core
- `src/runtime`
  runtime gateway, actuator apply, diagnostics
- `src/perception`
  detector, tracker, depth, observation processing
- `src/memory`
  working, episodic, semantic memory 계층
- `src/services`
  task policy와 memory-facing service
- `src/inference`
  detector, nav, VLM, training entrypoint
- `src/dashboard_backend`, `src/webrtc`
  운영 및 관찰 surface
- `src/ipc`
  in-process, ZMQ, shared memory transport
- `scripts`
  Windows 기준 주요 런처
- `docs/architecture`
  canonical architecture 설명
- `tests`
  계층별 검증 코드

## 참고 문서

- [docs/README.md](./docs/README.md)
- [docs/architecture/AURA_RUNTIME_ARCHITECTURE.md](./docs/architecture/AURA_RUNTIME_ARCHITECTURE.md)
- [docs/architecture/STATE_AND_OBSERVABILITY.md](./docs/architecture/STATE_AND_OBSERVABILITY.md)
- [docs/architecture/WORKER_CONTRACTS_AND_RECOVERY.md](./docs/architecture/WORKER_CONTRACTS_AND_RECOVERY.md)
