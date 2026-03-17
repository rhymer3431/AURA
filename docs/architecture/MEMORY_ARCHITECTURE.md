# Memory Architecture

AURA의 메모리는 이제 `WorldModelModule` 아래 read/write façade로 읽는 것이 기준이다. 기존 spatial / temporal / episodic / semantic memory는 유지하고, 그 위에 scratchpad와 keyframe bank를 얹어 planning backend 앞단에서 memory-aware planning을 지원한다.

## 메모리 계층

### 1. Scratchpad

현재 dual/interactive task의 짧은 작업 상태다.

- instruction
- planner mode
- task state (`pending`, `active`, `completed`, `cancelled`, `failed`, `idle`)
- checked locations
- recent hint
- next priority

이 scratchpad는 free-form 장문 로그가 아니라, 다음 S2 decision에 도움이 되는 짧은 상태 요약만 유지한다.

### 2. Object / Place DB

기존 `SpatialMemoryStore`, `TemporalMemoryStore`, `SemanticMemoryStore`가 그대로 main memory 역할을 한다.

- 무엇을 봤는가
- 어디서 봤는가
- 언제 봤는가
- 어떤 semantic hint가 붙는가

현재 frame write에서는 observation마다 아래 정보가 object metadata에 보강된다.

- `memory_summary`
- `room_id`
- `bearing_deg`
- `last_bbox_xyxy`
- `keyframe_id`
- optional `keyframe_crop_path`

### 3. Keyframe Bank

video memory 대신 대표 이미지 몇 장만 저장한다.

- 저장 위치: `state/memory/keyframes/`
- 저장 단위: `KeyframeRecord`
- 포함 정보
  - full image path
  - crop paths
  - summary
  - timestamp
  - robot pose / yaw
  - room id
  - focus labels / focus object ids

현재 저장은 heuristic 기반이다. 아래 경우에 keyframe 후보가 된다.

- 첫 scene capture
- room change
- 충분한 pose / yaw 변화
- 새 object 발견
- 현재 task instruction과 맞는 target 발견
- 마지막 keyframe 이후 시간이 충분히 지난 경우

## 업데이트 경로

메모리 업데이트의 canonical ingress는 `WorldModelModule.update(...)`이며, 현재 compatibility path에서는 `Supervisor.update_world_model(...)`와 `Supervisor.process_frame(...)`가 그 façade를 감싼다.

1. perception이 `ObsObject` 목록을 만든다.
2. `MemoryWritePath.record_perception_frame(...)`가 내부적으로 `MemoryService.record_perception_frame(...)`를 호출한다.
3. 이 함수 안에서
   - spatial association
   - temporal remember
   - episode observation 기록
   - keyframe 저장 여부 판단
   - object memory summary 갱신
   - scratchpad 보강
   가 한 번에 수행된다.
4. 이후 같은 observation은 `MissionModule.update(...)`를 거쳐 `TaskOrchestrator.on_observations()`로 넘어가지만, 메모리에 다시 쓰지는 않는다.

즉, frame-aware memory write는 world model, task-state consumption은 mission module이 맡는다.

## 조회 경로

dual / interactive task-active 상태에서만 memory retrieval을 켠다.

`MemoryReadPath.build_memory_context(...)`는 내부적으로 `MemoryService.build_memory_context(...)`를 호출하고 instruction을 세 종류로 분해한다.

- semantic query
  - 무엇을 찾는가
- spatial query
  - 어느 room / left-right / support clue와 관련 있는가
- temporal query
  - 최근성, 이전 관측 여부

현재 score는 다음 성분을 합친 heuristic이다.

- semantic score
- spatial score
- temporal score
- recency bonus
- confidence bonus
- optional proximity bonus

결과는 `MemoryContextBundle`로 묶인다.

- `ScratchpadState`
- `RetrievedMemoryLine` 3~5개
- `KeyframeRecord` 1~2개
- optional crop 1개
- backend seam용 `latent_backend_hint`

## Dual-System과의 연결

메모리 조회 결과는 planning backend의 System 2 경로에만 직접 들어간다.

- `NavigationRuntime`가 current frame 처리 직후 `world_model.memory_read.build_memory_context(...)`를 호출한다.
- 결과는 `ExecutionObservation.memory_context`에 붙는다.
- `DualPlannerInput` -> dual HTTP -> `PlanningCoordinator`/`DualOrchestrator` -> `System2Session` 순서로 그대로 전달된다.
- S1/NavDP 쪽 `sensor_meta`에는 memory payload를 섞지 않는다.

즉, 현재 구조는

- S2: memory-aware planner
- S1: unchanged trajectory/action generator

로 고정되어 있다.

## 지속성과 현재 한계

- SQLite snapshot에는 places / objects / semantic rules와 함께 scratchpad, keyframe metadata도 들어간다.
- keyframe 이미지 자체는 파일 시스템에 저장되고, snapshot에는 메타데이터만 남는다.
- retrieval은 대규모 embedding search가 아니라 현재 메모리 구조와 metadata를 이용한 lightweight heuristic search다.
- video timeline 전체를 다시 넣는 구조는 없다. visual history는 selected keyframe bank로 대체한다.
- `MemoryService`는 여전히 compatibility implementation이지만, 새 ownership 설명에서는 world-model state holder로 다룬다.
