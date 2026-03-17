# Dual-System Pipeline

이 문서는 현재 저장소에서 구현된 memory-aware dual-system planning 경로를 설명한다. 대상은 `scripts/powershell/run_aura_runtime.ps1`로 실행하는 G1 런타임이며, 구현 기준은 현재 `llama.cpp` dual-server 경로다.

## 핵심 원칙

- 메모리는 System 2 앞단에만 직접 들어간다.
- System 1/NavDP 인터페이스는 그대로 유지한다.
- video memory는 쓰지 않고 keyframe memory로 대체한다.
- System 2 출력 공간은 계속 `<y>, <x>` / `STOP` / `←` / `→` / `↓`로 고정한다.
- 현재 단계는 `2-stage hybrid`다.
  - System 2: `llama.cpp` 기반 memory-aware planner
  - System 1: 기존 `pixelgoal_step` 기반 trajectory generator
  - latent handoff는 이후 Torch/HF backend seam으로 남겨둔다.

## 현재 데이터 흐름

```
[RGB frame, depth, pose, instruction]
            |
            v
[Supervisor.process_frame]
- perception 실행
- structured memory write
- keyframe 선택
- scratchpad 갱신
            |
            v
[MemoryService]
- ScratchpadState
- Object/Place DB
- Keyframe Bank
            |
            v
[AuraRuntimeCommandSource]
- active dual task일 때 memory retrieval 수행
- MemoryContextBundle 생성
            |
            v
[PlanningSession / AsyncDualPlanner]
- observation + memory_context를 dual server로 전달
            |
            v
[DualOrchestrator / System 2]
- current image
- recent history image 1장
- retrieved text memory 3~5줄
- retrieved keyframe 1~2장
- optional crop 1장
            |
            +--------------------+
            |                    |
      STOP / yaw only        pixel goal
            |                    |
            v                    v
      planner-managed        NavDP pixelgoal_step
      control mode                |
                                  v
                           [System 1 / NavDP]
                           low-level trajectory
                                  |
                                  v
                                Execute
```

## 구성요소별 역할

### 1. `runtime.supervisor.Supervisor`

- perception 결과를 메모리에 한 번만 기록하는 canonical ingress다.
- `MemoryService.record_perception_frame(...)`를 호출해
  - object/place DB 갱신
  - keyframe bank 저장
  - scratchpad 보강
  을 수행한다.
- 이후 같은 observation을 `TaskOrchestrator`로 넘겨 follow/search/attention 같은 병렬 상태만 갱신한다.

### 2. `services.memory_service.MemoryService`

현재 dual path에서 VLM용 memory carrier 역할을 함께 가진다.

- `ScratchpadState`
  - 현재 dual/interactive task 상태
  - checked locations
  - 최근 힌트
  - 다음 우선순위
- object/place DB
  - 기존 spatial/temporal/semantic memory 재사용
- `KeyframeRecord`
  - 대표 장면 이미지와 object crop
- `build_memory_context(...)`
  - instruction을 semantic / spatial / temporal query로 분해
  - top-k memory line과 keyframe을 묶어 `MemoryContextBundle` 생성

### 3. `runtime.aura_runtime.AuraRuntimeCommandSource`

- `Supervisor.process_frame(...)` 직후 active dual task가 있으면 memory retrieval을 수행한다.
- retrieval 결과를 `ExecutionObservation.memory_context`에 붙인다.
- interactive lifecycle과 scratchpad를 동기화한다.
  - instruction submit -> `pending`
  - dual/interactive task start -> `active`
  - cancel/complete/failure -> clear

### 4. `runtime.planning_session.PlanningSession`

- `ExecutionObservation`에 `memory_context`를 포함한다.
- `DualPlannerInput`으로 memory context를 그대로 넘긴다.
- point-goal/no-goal 경로는 건드리지 않는다.

### 5. `services.dual_orchestrator.DualOrchestrator`

- `memory_context`는 System 2 request preparation에만 사용한다.
- System 1 호출에는 넣지 않는다.
- 즉, 메모리가 S1에 미치는 영향은
  - 더 좋은 System 2 decision
  - 더 좋은 pixel goal
  - 그 결과로 더 좋은 trajectory
  의 간접 효과만 허용한다.

## System 2 입력 형식

현재 System 2 request body는 다음 요소를 포함한다.

- 현재 observation image
- recent history image 1장
- scratchpad text
- retrieved memory text 3~5줄
- retrieved keyframe 1~2장
- optional crop 1장
- 기존 event flags (`force_s2`, `stuck`, `collision_risk`)

출력은 계속 아래 다섯 경우만 허용한다.

- `<y>, <x>`
- `STOP`
- `←`
- `→`
- `↓`

## System 1 보존 범위

현재 System 1 실행은 그대로 `pixelgoal_step` 호출이다.

- 입력
  - RGB
  - depth
  - pixel goal
  - sensor meta
  - camera pose
- 출력
  - world trajectory

`memory_context`는 HTTP contract 상 별도 필드이며, `sensor_meta`에 섞이지 않는다.

## 현재 구현 한계

- latent extraction / latent -> S1 handoff는 아직 없다.
- current dual server는 `llama.cpp` path 기준이다.
- keyframe selection은 heuristic 기반이다.
- richer spatial language grounding은 현재 object metadata, room id, bearing, recency를 이용한 lightweight retrieval 수준이다.
