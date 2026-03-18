# AURA Runtime Architecture

## 1. 목표

현재 G1 런타임의 canonical 구조는 다음 두 원칙으로 정리된다.

- 모든 write-side 판단과 상태 전이는 `MainControlServer`가 소유한다.
- 모든 read-side 관측은 `WorldStateSnapshot`을 통해 노출한다.

이 구조는 기존의 bridge / planning session / dual orchestration / executor가 나눠 들고 있던 상태와 정책을 중앙 서버 계층으로 이동시키는 것을 목표로 한다.

## 2. 현재 상위 구조

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

## 3. 핵심 컴포넌트

### 3.1 Runtime / Gateway

주요 파일:

- [aura_runtime.py](/mnt/c/Users/mango/project/isaac-aura/src/runtime/aura_runtime.py)
- [frame_bridge_runtime.py](/mnt/c/Users/mango/project/isaac-aura/src/runtime/frame_bridge_runtime.py)

역할:

- frame capture
- task/control ingress
- `MainControlServer.tick()` 호출
- 최종 actuator command 적용
- telemetry publish

중요한 점:

- runtime은 더 이상 planner truth source가 아니다.
- frame bridge는 gateway 역할만 하며 planner state owner가 아니다.

### 3.2 Main Control Server

주요 파일:

- [main_control_server.py](/mnt/c/Users/mango/project/isaac-aura/src/server/main_control_server.py)

역할:

- frame/task/runtime 상태를 한 곳에서 ingest
- planning 전 pre-plan 정책 판단
- perception/memory enrichment orchestration
- planner/nav/locomotion 실행
- planning outcome 평가
- safety event 적용
- final command arbitration
- canonical state commit

`tick()`의 개념적 순서는 다음과 같다.

1. frame/task ingest
2. current recovery state 조회
3. `DecisionEngine` pre-plan directive 계산
4. perception/memory enrichment
5. planner/nav/locomotion 실행
6. planning outcome 기반 recovery transition
7. safety transition 적용
8. `WorldStateStore` commit
9. `CommandResolver` final command 생성
10. task sync + snapshot publish

### 3.3 Task Manager

주요 파일:

- [task_manager.py](/mnt/c/Users/mango/project/isaac-aura/src/server/task_manager.py)

역할:

- planner mode lifecycle 관리
- interactive / pointgoal / dual task 상태 소유
- dashboard task request / cancel 처리
- task completion / failure / clear 시 state reset 연계

중요한 점:

- `PlanningSession`은 task lifecycle owner가 아니다.
- interactive command id, task state, task reset은 `TaskManager` 기준이다.

### 3.4 Planner Coordinator

주요 파일:

- [planner_coordinator.py](/mnt/c/Users/mango/project/isaac-aura/src/server/planner_coordinator.py)
- [planner_runtime_engine.py](/mnt/c/Users/mango/project/isaac-aura/src/server/planner_runtime_engine.py)
- [planner_runtime_state.py](/mnt/c/Users/mango/project/isaac-aura/src/server/planner_runtime_state.py)

역할:

- observation enrichment
- planning context 구성
- nav / locomotion request 발행
- worker result validation
- planner runtime cache와 planner-facing sequencing 관리

중요한 점:

- `PlanningSession`에 있던 canonical planner cache는 여기와 `PlannerRuntimeState`로 이동했다.
- stale/mismatch discard는 coordinator 중심으로 처리된다.

### 3.5 Decision / Safety / Command

주요 파일:

- [decision_engine.py](/mnt/c/Users/mango/project/isaac-aura/src/server/decision_engine.py)
- [safety_supervisor.py](/mnt/c/Users/mango/project/isaac-aura/src/server/safety_supervisor.py)
- [command_resolver.py](/mnt/c/Users/mango/project/isaac-aura/src/server/command_resolver.py)

역할 분리:

- `DecisionEngine`
  - retrieve/replan/retry/backoff/S2 requery 정책
- `SafetySupervisor`
  - sensor missing, stale, timeout, safe-stop 관련 safety event 생성
- `CommandResolver`
  - recovery state와 proposal을 최종 command/status로 매핑

중요한 점:

- `CommandResolver`는 정책 owner가 아니다.
- `SafetySupervisor`는 최종 command를 직접 mutate하지 않는다.

## 4. Legacy 컴포넌트의 현재 위치

### 4.1 PlanningSession

주요 파일:

- [planning_session.py](/mnt/c/Users/mango/project/isaac-aura/src/runtime/planning_session.py)

현재 역할:

- sensor/bootstrap transport
- planner adapter access
- lightweight helper

더 이상 하지 않는 것:

- task lifecycle ownership
- planner cache ownership
- last-goal / last-s2 / active-plan canonical state ownership

### 4.2 DualOrchestrator

현재 상태:

- 삭제됨

대체:

- 정책은 `DecisionEngine`
- planner wiring은 `PlannerCoordinator`
- dual HTTP path는 [dual_planner_service.py](/mnt/c/Users/mango/project/isaac-aura/src/server/dual_planner_service.py) 기반

### 4.3 SubgoalExecutor

주요 파일:

- [subgoal_executor.py](/mnt/c/Users/mango/project/isaac-aura/src/runtime/subgoal_executor.py)
- [worker.py](/mnt/c/Users/mango/project/isaac-aura/src/locomotion/worker.py)

현재 상태:

- low-level proposal 계산은 locomotion worker로 이동
- final status/stop/failure 판단은 server 계층으로 이동
- subgoal executor는 compatibility wrapper 수준

## 5. 현재 구조에서 중요한 금지 규칙

- runtime local state를 planner truth source로 쓰지 않는다.
- `PlanningSession`에서 task/plan/recovery truth를 읽지 않는다.
- dashboard/WebRTC가 planner overlay나 legacy wrapper 상태를 직접 읽지 않는다.
- worker가 final task status를 결정하지 않는다.

## 6. 현재 구조의 범위

이 문서는 A-F 완료 상태를 설명한다.

아직 남아 있는 후속 범위:

- compatibility surface 축소
- `Local Stack` / `Memory Agent`의 server core 재사용 경로 정리
- 문서/guardrail/dead code cleanup
