# Migration Summary A To F

## A. PlanningSession shadow ownership 제거

핵심 변화:

- `PlanningSession`에서 task lifecycle과 planner canonical state를 걷어냄
- planner cache, goal/traj state, global-route state를 server 계층으로 이동

결과:

- `PlanningSession`은 transport/bootstrap/helper 역할만 남음
- 중앙 서버 없이 planner truth를 재구성할 수 없게 됨

## B. Dual orchestration 해체

핵심 변화:

- legacy `DualOrchestrator` 삭제
- 정책은 `DecisionEngine`
- request/result wiring은 `PlannerCoordinator`
- dual HTTP path는 `DualPlannerService`로 정리

결과:

- dual policy와 G1 policy가 같은 server 계층 모델로 이동
- orchestration giant object가 사라짐

## C. Executor 분해

핵심 변화:

- low-level locomotion proposal 계산을 worker로 이동
- final stop/fail/stale/status 판단을 server 계층으로 이동

결과:

- locomotion은 proposal producer
- server는 final arbiter

## D. Snapshot read model 정리

핵심 변화:

- `WorldStateSnapshot` typed tree 도입
- `SnapshotAdapter`를 유일한 read-model translation layer로 고정
- dashboard/WebRTC/runtime mirror가 snapshot만 읽도록 변경

결과:

- observability truth source가 하나로 정리됨
- runtime locals와 UI state divergence를 줄임

## E. Worker/client contract 강화

핵심 변화:

- typed worker request/result schema 추가
- metadata stamp/validation/timeout/mismatch discard 공통화
- planner coordinator가 raw runtime object에 직접 의존하지 않도록 조정

결과:

- same-process adapter이지만 RPC 가능한 형태의 contract를 확보
- stale/mismatch/timeout semantics가 일관됨

## F. Recovery state machine 승격

핵심 변화:

- fallback/retry/safe-stop/stale/sensor wait 분기를 typed recovery state machine으로 승격
- recovery semantics를 snapshot과 mirror에 반영
- 역할을 decision/safety/command/store로 분리

결과:

- 장애 대응이 분기 모음이 아니라 상태 전이로 표현됨
- dashboard/WebRTC/runtime mirror가 같은 recovery truth를 공유함

## A-F 전체 효과

구조 전환 이후 얻은 효과:

- 중앙 서버가 실제 write-side owner가 됨
- `WorldStateSnapshot`이 실제 read-side owner가 됨
- planner/session/orchestrator/executor의 중복 ownership이 줄어듦
- observability와 recovery semantics가 일관됨
- worker 분리와 다른 실행 surface 재사용을 위한 기반이 마련됨

## 현재 남은 과제

아직 남은 후속 범위:

- G: compatibility surface 축소와 server-core 재사용 경로 정리
- H: architecture docs/guardrails/dead code cleanup

즉, A-F는 중앙 서버 구조를 실제로 성립시키는 단계였고, 이후 단계는 그 구조를 더 얇고 공용화된 형태로 다듬는 단계다.
