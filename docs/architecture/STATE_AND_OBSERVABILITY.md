# State And Observability

## 1. 상태 모델 개요

현재 AURA 런타임은 상태를 두 층으로 나눈다.

- write-side canonical state: `WorldStateStore`
- read-side canonical state: `WorldStateSnapshot`

주요 파일:

- [world_state_store.py](/mnt/c/Users/mango/project/isaac-aura/src/server/world_state_store.py)
- [world_state.py](/mnt/c/Users/mango/project/isaac-aura/src/schemas/world_state.py)
- [snapshot_adapter.py](/mnt/c/Users/mango/project/isaac-aura/src/server/snapshot_adapter.py)

## 2. Write-side canonical state

`WorldStateStore`가 canonical하게 보유하는 정보:

- current task
- mode
- robot pose / frame / sensor meta
- last perception summary
- last memory context summary
- planning result and nav plan versions
- execution result and last command decision
- safety booleans
- recovery state
- runtime compatibility config

중요한 점:

- 이 상태는 runtime locals에 분산 보관하지 않는다.
- planner/session/bridge는 이 상태의 owner가 아니다.

## 3. Read-side canonical state

`WorldStateSnapshot`는 다음 섹션으로 나뉜 typed dataclass tree다.

- `task`
- `mode`
- `robot`
- `perception`
- `memory`
- `planning`
- `execution`
- `safety`
- `runtime`

이 snapshot은 다음 요구를 만족해야 한다.

- dashboard가 직접 읽을 수 있어야 함
- WebRTC state mirror가 직접 읽을 수 있어야 함
- legacy runtime payload를 mechanical하게 생성할 수 있어야 함
- 직렬화/역직렬화가 가능해야 함

## 4. Observability source of truth

현재 observability truth source는 `WorldStateSnapshot` 하나다.

즉, 다음은 truth source가 아니다.

- `PlanningSession`
- runtime local cached planner fields
- `planner_overlay`
- compatibility executor/session wrapper

## 5. Snapshot adapter의 역할

`SnapshotAdapter`는 snapshot 자체를 소유하지 않는다.

역할은 변환만이다.

- dashboard state payload 생성
- WebRTC `snapshot` payload 생성
- WebRTC `frame_meta` payload 생성
- legacy runtime compatibility payload 생성

원칙:

- 필드명/shape 호환이 필요하면 adapter에서만 변환한다.
- UI/backend는 내부 객체를 뒤져서 planner truth를 재조립하지 않는다.

## 6. Dashboard / WebRTC / Runtime mirror

### 6.1 Dashboard

주요 파일:

- [state.py](/mnt/c/Users/mango/project/isaac-aura/src/dashboard_backend/state.py)

현재 dashboard state 구성:

- `WorldStateSnapshot` 기반 runtime/sensor/perception/memory 상태
- process/session/log/service 상태는 dashboard backend가 별도로 병합

중요한 점:

- dashboard는 snapshot 밖에서 planner state를 합성하지 않는다.

### 6.2 WebRTC

주요 파일:

- [subscriber.py](/mnt/c/Users/mango/project/isaac-aura/src/webrtc/subscriber.py)

현재 WebRTC path:

- `HealthPing(component="aura_runtime")`에서 `worldState` 수신
- `WorldStateSnapshot.from_dict(...)`로 복원
- `SnapshotAdapter`로 state/meta 생성

중요한 점:

- `planner_overlay`는 visual geometry용일 뿐 planner truth source가 아니다.

### 6.3 Runtime compatibility mirror

주요 파일:

- [aura_runtime.py](/mnt/c/Users/mango/project/isaac-aura/src/runtime/aura_runtime.py)

현재 runtime mirror:

- `details["worldState"]`: serialized `WorldStateSnapshot`
- `details["snapshot"]`: legacy mirror payload

즉, legacy runtime payload도 이제 snapshot의 read-only mirror다.

## 7. Recovery semantics의 observability 반영

Recovery 의미는 snapshot과 mirror에서 동일해야 한다.

반영 위치:

- `WorldStateSnapshot.safety.recovery_state`
- runtime legacy planner payload의 recovery fields
- dashboard runtime/sensor sections
- WebRTC `snapshot` / `frame_meta`

표현 의미:

- `safeStop`
- `stale`
- `timeout`
- `sensorUnavailable`
- `recoveryState`
- `recoveryEnteredAtNs`
- `recoveryRetryCount`
- `recoveryBackoffUntilNs`
- `recoveryReason`

## 8. 현재 상태 dump의 의미

현재 구조에서는 `WorldStateSnapshot` 하나를 dump하면 다음을 동시에 재현할 수 있어야 한다.

- 현재 task 상태
- planner/nav progress
- last command decision
- sensor health
- recovery semantics
- dashboard와 WebRTC가 보여야 할 핵심 상태
