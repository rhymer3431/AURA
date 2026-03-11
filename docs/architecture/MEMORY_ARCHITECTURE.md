# Memory Architecture

AURA의 메모리는 단순 캐시가 아니라, perception 결과를 구조화된 상태로 누적하고 태스크 회상과 recovery에 재사용하는 런타임 서비스다. 현재 구조에서는 `services.memory_service.MemoryService`가 메모리 계층의 중심이며, `TaskOrchestrator`와 여러 서비스가 이를 공유한다.

## 핵심 구성요소

현재 메모리 아키텍처의 중심 구성요소는 다음과 같다.

- `MemoryService`
  - 메모리 저장소, query, consolidation, persistence를 묶는 상위 서비스
- `SpatialMemoryStore`
  - `PlaceNode`, `ObjectNode` 중심의 공간 기억
- `TemporalMemoryStore`
  - 시간순 사건, follow loss, speaker binding, 재획득 힌트 저장
- `EpisodicMemoryStore`
  - 태스크 단위 episode record 저장
- `SemanticMemoryStore`
  - 반복 episode에서 얻은 rule-like 힌트 저장
- `WorkingMemory`
  - 현재 태스크에 의미 있는 후보를 우선순위화
- `MemoryQueryEngine`
  - recall 시 spatial/semantic/working memory를 결합
- `MemoryConsolidator`
  - episode를 semantic rule로 통합
- `SemanticConsolidationService`
  - episode summary와 semantic 보강을 수행
- `SQLiteMemoryPersistence`
  - snapshot 저장이 필요할 때 SQLite에 직렬화

## 각 저장소가 담당하는 기억

### Spatial Memory

`SpatialMemoryStore`는 관측된 객체와 장소의 관계를 유지한다.

- 객체 observation을 기존 object/place와 연관짓는다.
- object -> place -> navigation target 형태의 회상을 가능하게 한다.
- 이후 `"아까 봤던 사과"` 같은 질의가 들어오면 recall 출발점이 된다.

### Temporal Memory

`TemporalMemoryStore`는 시간에 민감한 사건을 다룬다.

- speaker event
- speaker binding
- follow target 상실/재획득 단서
- track 기반 최근성 정보

이 계층은 follow recovery나 발화자 주시 같은 태스크에서 직접적으로 쓰인다.

### Episodic Memory

`EpisodicMemoryStore`는 태스크 단위 record를 쌓는다.

- 원본 command text
- intent
- target metadata
- 본 객체/장소
- candidate attempt
- recovery action
- success/failure와 summary

즉, “무엇을 하려 했고 어떤 후보를 거쳤으며 왜 실패/성공했는가”를 구조적으로 남기는 역할이다.

### Semantic Memory

`SemanticMemoryStore`는 반복된 episode나 bootstrap rule에서 얻은 규칙성 힌트를 저장한다.

- 특정 intent/target_class/room 조합에 대한 선호
- 성공률/지원 횟수
- planner hint metadata

현재 구현은 rule/template 중심이며, 대규모 symbolic reasoner는 아니다.

### Working Memory

`WorkingMemory`는 장기 저장소 자체가 아니라, 현재 질의와 상황에서 가장 유의미한 후보를 고르는 선택 계층이다.

- recency
- reachability
- confidence
- semantic bonus

이 기준으로 recall candidate를 정렬해 현재 태스크가 바로 사용할 수 있는 active subset을 만든다.

## 현재 업데이트 흐름

메모리 업데이트는 perception과 task orchestration에 직접 연결되어 있다.

1. `Supervisor.process_frame()`가 RGB/depth를 `PerceptionPipeline`에 전달한다.
2. pipeline이 object observation과 speaker event를 생성한다.
3. `TaskOrchestrator.on_observations()`가 person track과 attention 상태를 갱신한다.
4. 같은 함수 안에서 `MemoryService.observe_objects()`가 호출된다.
5. 각 observation은 다음 순서로 저장된다.
   - spatial association
   - temporal remember
   - active episode에 place/object 기록
6. speaker binding이 확인되면 temporal memory와 active episode가 함께 보강된다.

즉, perception 결과는 별도 후처리 큐가 아니라 orchestration 루프 안에서 즉시 구조화 메모리로 들어간다.

## `MemoryService`가 직접 제공하는 주요 기능

현재 `MemoryService`는 단순 CRUD보다 넓은 책임을 가진다.

- `observe_objects(...)`
  - object observation을 spatial/temporal/episode에 반영
- `record_speaker_event(...)`
  - 발화 방향 이벤트 저장
- `record_speaker_binding(...)`
  - speaker와 person/track 연결
- `record_follow_target(...)`
  - 현재 follow target을 active episode에 기록
- `record_candidate_attempt(...)`
  - recall 후보와 적용 semantic rule 추적
- `record_recovery_action(...)`
  - recovery 단계 기록
- `start_episode(...)`, `finish_episode(...)`
  - 태스크 시작/종료 관리
- `recall_object(...)`
  - query engine을 통해 회상 수행
- `reacquire_follow_target(...)`
  - temporal memory 기반 target 재획득
- `persist_snapshot(...)`
  - places / objects / semantic rules를 SQLite snapshot으로 저장

## 회상과 태스크 서비스의 연결

메모리는 단독으로 존재하지 않고 `TaskOrchestrator`와 그 하위 서비스에서 직접 사용된다.

### 기억 기반 객체 탐색

`GO_TO_REMEMBERED_OBJECT` 상태에서는 `ObjectSearchService`가 `MemoryService.recall_object(...)`를 호출한다.

현재 회상 과정은 다음 정보를 결합한다.

- spatial memory의 object 후보
- semantic memory의 rule
- working memory의 우선순위화
- 현재 로봇 pose
- room/context 정보

그 결과 가장 유력한 object/place가 선택되고, 이후 탐색/접근 명령 생성으로 이어진다.

### Follow recovery

follow target을 잃으면 recovery 경로가 temporal memory와 semantic hint를 참고한다.

- 최근 track 재획득
- cone/local search
- recovery action 기록

이 흐름은 “최근에 어디 있었는가”와 “어떤 회복 전략이 시도되었는가”를 메모리에 남긴다.

### Attend / caller binding

speaker event와 person tracker가 결합되면 temporal memory에 speaker binding이 기록된다. 이 정보는 caller 주시, follow target 식별, 이후 episode summary에 영향을 준다.

## 통합과 지속성

episode가 끝나면 메모리는 단순 종료되지 않는다.

1. `finish_episode(...)`가 success/failure와 summary를 기록한다.
2. `SemanticConsolidationService.summarize_episode(...)`가 episode 요약을 만든다.
3. `MemoryConsolidator.consolidate_episode(...)`가 semantic rule 업데이트를 수행한다.
4. 이후 유사 태스크 recall에서 semantic hint가 다시 사용된다.

지속성이 필요한 경우 `SQLiteMemoryPersistence`가 snapshot을 저장한다. 현재 snapshot payload는 전체 temporal/episodic history가 아니라 다음에 집중한다.

- places
- objects
- semantic rules

또한 `MemoryAgentRuntime`는 초기화 시 bootstrap semantic rule을 하나 주입해 기본 탐색 힌트를 확보한다.

## Live Smoke와의 관계

Live Smoke는 메모리 품질을 평가하는 벤치마크가 아니라, 메모리 ingress가 실제로 연결되어 있는지 검증하는 경로다.

현재 smoke tier는 다음처럼 구분된다.

- `sensor_smoke_pass`
  - 센서/D455/frame ingress 검증
- `pipeline_smoke_pass`
  - perception pipeline 도달 검증
- `memory_smoke_pass`
  - `MemoryService` 업데이트 경로 실행 검증

빈 장면이라면 `pipeline`까지는 통과하고 `memory`는 미통과일 수 있다. 이것은 센서 장애가 아니라 “메모리에 넣을 detection이 없었다”는 뜻이다.

## 현재 한계

- semantic consolidation은 아직 rule/template 중심이다.
- working memory는 heuristic ranking 계층이지 범용 추론 엔진은 아니다.
- SQLite snapshot은 전체 장기 메모리 상태를 완전 재구성하는 포맷이 아니라, 핵심 객체/장소/rule 보존에 가깝다.
- memory tier 검증은 detector 품질이 아니라 ingress wiring 확인에 초점이 있다.
