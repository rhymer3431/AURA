# Worker Contracts And Recovery

## 1. Worker boundary

현재 worker boundary는 same-process adapter이지만, interface는 분리 프로세스 가능 형태로 고정돼 있다.

주요 파일:

- [workers.py](/mnt/c/Users/mango/project/isaac-aura/src/schemas/workers.py)
- [worker_clients.py](/mnt/c/Users/mango/project/isaac-aura/src/clients/worker_clients.py)
- [planner_coordinator.py](/mnt/c/Users/mango/project/isaac-aura/src/server/planner_coordinator.py)

## 2. Typed request/result contract

현재 정의된 주요 worker contract:

- `PerceptionRequest` / `PerceptionResult`
- `MemoryRequest` / `MemoryResult`
- `NavRequest` / `NavResult`
- `LocomotionRequest` / `LocomotionResult`
- `S2Request` / `S2Result`

공통 메타:

- `trace_id`
- `task_id`
- `frame_id`
- `timestamp_ns`
- `source`
- `timeout_ms`
- `plan_version`
- `goal_version`
- `traj_version`

## 3. Validation rule

공통 validation helper는 다음 역할을 한다.

- request stamping
- metadata inheritance
- timeout detection
- frame/task/version mismatch rejection
- normalized timeout/error/discard result 생성

중요한 점:

- stale/mismatch discard는 중앙 server 계층에서 수행한다.
- worker는 자기 결과가 채택될지 최종 결정하지 않는다.

## 4. PlannerCoordinator와 worker 사용 방식

`PlannerCoordinator`는 raw runtime object를 아무렇게나 넘기지 않고 request/result schema를 경유한다.

흐름:

1. frame/task context에서 metadata stamp
2. typed request 생성
3. worker client 호출
4. `finalize_worker_result(...)`로 timeout/mismatch 검증
5. 유효한 결과만 planning path에 반영

## 5. Locomotion boundary

low-level locomotion은 proposal만 계산한다.

주요 파일:

- [worker.py](/mnt/c/Users/mango/project/isaac-aura/src/locomotion/worker.py)
- [command_resolver.py](/mnt/c/Users/mango/project/isaac-aura/src/server/command_resolver.py)

원칙:

- locomotion worker는 task lifecycle을 모른다.
- locomotion worker는 final success/failure를 결정하지 않는다.
- final status는 server 계층에서만 결정된다.

## 6. Recovery state machine

주요 파일:

- [recovery.py](/mnt/c/Users/mango/project/isaac-aura/src/schemas/recovery.py)
- [recovery_state.py](/mnt/c/Users/mango/project/isaac-aura/src/server/recovery_state.py)
- [decision_engine.py](/mnt/c/Users/mango/project/isaac-aura/src/server/decision_engine.py)
- [safety_supervisor.py](/mnt/c/Users/mango/project/isaac-aura/src/server/safety_supervisor.py)
- [command_resolver.py](/mnt/c/Users/mango/project/isaac-aura/src/server/command_resolver.py)

현재 recovery state:

- `NORMAL`
- `REPLAN_PENDING`
- `WAIT_SENSOR`
- `SAFE_STOP`
- `RECOVERY_TURN`
- `FAILED`

canonical field:

- `current_state`
- `entered_at_ns`
- `retry_count`
- `backoff_until_ns`
- `last_trigger_reason`

## 7. Recovery ownership

책임 분리:

- `DecisionEngine`
  - replanning / retry / backoff / S2 requery 정책
  - planning outcome 기반 transition
- `SafetySupervisor`
  - sensor missing / timeout / stale trajectory 기반 transition
  - safety override intent 계산
- `CommandResolver`
  - recovery state를 final command/status로 매핑
- `WorldStateStore`
  - recovery state canonical 저장과 snapshot 반영

## 8. 주요 transition

현재 기준의 명시적 전이:

- `sensor_missing`
  - `WAIT_SENSOR` 또는 `SAFE_STOP`
- `sensor_restored`
  - `WAIT_SENSOR -> NORMAL`
- `trajectory_stale`
  - `REPLAN_PENDING`
- `planning_failure`
  - `REPLAN_PENDING` 유지 또는 `RECOVERY_TURN`
  - budget 초과 시 `FAILED`
- `planning_success`
  - `NORMAL`
- `timeout`
  - `SAFE_STOP`
- `task_reset`
  - `NORMAL`

## 9. Config-backed policy

Recovery policy는 args 기반으로 설정된다.

대표 항목:

- `max_stale_age_ms`
- `planner_retry_budget`
- `safe_stop_timeout_ms`
- `sensor_wait_budget_ms`
- `recovery_turn_retry_limit`
- `s2_retry_backoff_ms`

의미:

- 마법값을 줄이고 테스트 가능한 정책으로 만든다.
- G1 외 surface로 확장할 때도 같은 policy model을 재사용할 수 있게 한다.

## 10. 현재 한계

현재 recovery 구조는 G1 `MainControlServer` 경로에 우선 적용된 상태다.

아직 남은 범위:

- dual/local stack/memory agent 경로와의 완전한 shared policy 정렬
- compatibility surface 축소
- 문서/guardrail/dead code cleanup
