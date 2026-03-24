# System2 TALK/NAV Router + LoRA Design

## 1. 목표

이 문서는 현재 `isaac-aura` 코드베이스를 기준으로, 단일 VLM 백본을 `TALK`와 `NAV` 두 용도로 함께 사용하기 위한 프롬프트/데이터셋/학습 설계를 정리한다.

핵심 목표는 다음과 같다.

- 기본 모드는 `TALK`로 둔다.
- 라우팅도 같은 VLM 백본으로 수행하되, 출력은 `TALK` 또는 `NAV` 한 토큰으로 제한한다.
- `NAV`로 라우팅된 뒤에는 별도 시스템 프롬프트를 주입해 사용자 응답과 내부 계획을 생성한다.
- 실제 이동 중 low-level 의사결정은 현재 `System2Session` 계약을 유지한다.
- 작업 완료 뒤에는 다시 `TALK` 모드로 복귀해 결과를 사용자에게 보고한다.

즉, "단일 VLM + 단일 LoRA"를 유지하되, 추론 시점에는 `route -> TALK/NAV prompt injection -> NAV step loop -> TALK report`로 나누어 다룬다.

## 2. 현재 코드 기준 확인된 사실

### 2.1 현재 모드 분류는 규칙 기반이다

- `src/server/execution_mode_classifier.py`
  - 현재 `TALK`, `NAV`, `MEM_NAV`, `EXPLORE` 분류는 키워드 규칙 기반이다.
  - 기본 fallback은 `TALK`다.

### 2.2 현재 `TALK`는 실제 VLM 응답 생성까지 이어지지 않는다

- `src/server/task_manager.py`
  - `TALK`로 분류되면 `hookStatus: ready`만 세팅하고 planner task만 기록한다.
  - 현재는 실제 대화 응답을 생성하는 VLM 단계가 없다.

### 2.3 현재 System2는 순수 네비게이션 결정기다

- `src/inference/vlm/system2_session.py`
  - system prompt는 `"You must return only a single navigation decision string."`
  - 허용 출력은 다음 다섯 가지뿐이다.
    - `"<y>, <x>"`
    - `STOP`
    - `←`
    - `→`
    - `↓`
  - `parse_system2_output()`도 위 계약을 전제로 구현돼 있다.

### 2.4 현재 LoRA seed dataset도 navigation-only다

- `src/inference/training/system2_memory_lora.py`
- `artifacts/datasets/system2_memory_lora_seed/README.md`
  - 현재 seed set은 `direct_pixel`, `memory_pixel`, `memory_turn_left`, `memory_turn_right`, `stop`, `wait`만 포함한다.
  - 즉, 현재 데이터셋은 `NAV step`만 가르치고 `TALK`, `route`, `report`는 포함하지 않는다.

### 2.5 프롬프트 로직은 Python 쪽에서 잡는 것이 맞다

- `scripts/run_system.ps1`
  - llama.cpp용 `--chat-template-file` 경로를 잡고 있지만, 파일이 없으면 그냥 null로 처리한다.
- 현재 실제 요청 본문은 `System2Session._build_request_body()`에서 직접 구성된다.

따라서 TALK/NAV 분기 로직과 system prompt 주입 위치는 Jinja 템플릿 하나에 넣기보다, Python 런타임 계층에서 명시적으로 관리하는 편이 현재 구조와 맞다.

## 3. 제안하는 단일 VLM 런타임 계약

단일 백본을 쓰되, 호출 목적을 5종류로 나눈다.

1. `ROUTE`
   - 입력: 최근 대화 이력 + 현재 유저 발화 + 최소 컨텍스트
   - 출력: `TALK` 또는 `NAV`
   - 실패 시 fallback: `TALK`

2. `TALK`
   - 입력: 최근 대화 이력 + 현재 유저 발화 + 필요 시 task result summary
   - 출력: 사용자에게 바로 말할 한국어 응답 한 개

3. `NAV_START`
   - 입력: 유저 요청 + 메모리 컨텍스트 + 환경 상태 + user anchor
   - 출력:
     - 첫 줄: 사용자에게 말할 짧은 승인 문장
     - 이후 줄: 내부 실행 trace (`[목표]`, `[NAV]`, `[목표 완료]`)

4. `NAV_STEP`
   - 입력: 현재 프레임 + history + memory context + events
   - 출력: 현재 `System2Session`과 동일한 단일 navigation decision string

5. `TALK_REPORT`
   - 입력: `NAV` 작업 결과 요약 + 검증 결과
   - 출력: 사용자에게 전달할 최종 보고 문장

권장 상태 전이는 다음과 같다.

```text
default TALK
  -> ROUTE
    -> TALK        : 일반 대화
    -> NAV_START   : 실행 승인 + 내부 계획 생성
        -> NAV_STEP loop
        -> TALK_REPORT
        -> default TALK 복귀
```

## 4. 프롬프트 설계

### 4.1 ROUTE system prompt

```text
You are AURA's execution-mode router.
Your job is to choose whether the next turn should use TALK mode or NAV mode.

Rules:
- Default to TALK.
- Return NAV only when the user is asking for a physical action, movement, inspection, fetching, following, checking, or any embodied task that requires the robot to move or observe the environment.
- Return TALK for greetings, chitchat, explanations, questions, summaries, and follow-up conversation.
- If the request is ambiguous, return TALK.
- Return exactly one token: TALK or NAV.
Do not output JSON.
Do not output any extra words.
```

출력 예시:

- `안녕` -> `TALK`
- `거실에 가서 tv가 꺼졌는지 확인해줘` -> `NAV`

### 4.2 TALK system prompt

```text
You are AURA's conversation agent.
Speak natural, concise Korean.

Rules:
- Stay in conversation mode.
- Do not claim to have physically checked or moved unless task_result context is explicitly provided.
- If task_result is provided, report only grounded facts from that result.
- If the user greets you, respond briefly and politely.
- If the user asks a question that requires movement or inspection, do not execute it here; the router will send that request to NAV mode.
- Keep answers short, clear, and factual.
Do not output JSON unless the caller explicitly requests a structured format.
```

출력 예시:

- 입력: `안녕`
- 출력: `안녕하세요.`

최종 보고 예시:

- 입력 컨텍스트: `task_result = {"room":"거실","object":"tv","attribute":"power_state","value":"off"}`
- 출력: `거실에서 확인해본 결과, tv가 꺼져 있습니다.`

### 4.3 NAV system prompt

`NAV`는 phase를 반드시 입력받아야 한다. 그래야 같은 백본이 user-facing 승인과 low-level pixel decision을 섞어 내지 않는다.

권장 phase:

- `phase=start`
- `phase=step`

#### NAV prompt

```text
You are AURA's embodied request agent.
The current task has already been routed to NAV mode.

You must obey the phase field.

If phase=start:
- First line must be a short Korean acknowledgement to the user.
- After the first line, output only structured execution trace lines.
- Allowed trace lines:
  - [목표] <goal text>
  - [NAV] <action text>
  - [목표 완료]
- Use the trace to decompose the task into executable subgoals.
- If the task requires checking something and then reporting back to the user, include the return-to-user subgoal explicitly.

If phase=step:
- Output exactly one navigation decision string.
- Allowed outputs are only:
  - <y>, <x>
  - STOP
  - ←
  - →
  - ↓
- Do not output any explanation.
- Do not output JSON.

Ground all decisions in the provided observation, memory context, task context, and current subgoal.
```

#### NAV start 예시

입력 요청:

```text
거실에 가서 tv가 꺼졌는지 확인해줘.
```

권장 출력:

```text
지금 확인하고 올게요.
[목표] 거실에 가서 tv가 꺼졌는지 확인
[NAV] 거실로 이동
[NAV] tv가 꺼졌는지 확인
[목표 완료]
[목표] 유저에게 돌아가서 결과 보고
[NAV] 유저에게 이동
```

운영 규칙:

- 첫 줄만 사용자에게 그대로 발화한다.
- `[목표]`, `[NAV]`, `[목표 완료]` 줄은 내부 planner trace로 파싱한다.
- 결과 보고는 `TALK_REPORT` 단계에서 다시 TALK prompt로 생성한다.

#### NAV step 예시

```text
241, 156
```

또는

```text
STOP
```

## 5. 데이터셋 설계

### 5.1 기존 dataset을 확장하는 방향

현재 `system2_memory_lora` 스키마는 `NAV_STEP`에 이미 맞춰져 있으므로, 완전히 새 포맷을 만드는 것보다 기존 스키마를 확장하는 편이 안전하다.

권장 새 스키마 이름:

- `system2_talk_nav_lora_v1`

공통 필드:

- `schema_version`
- `sample_id`
- `split`
- `task_type`
- `instruction`
- `dialog_history`
- `events`
- `memory_context`
- `current_image`
- `history_images`
- `target_text`

권장 추가 필드:

- `route_label`
  - `TALK` 또는 `NAV`
- `phase`
  - `route`, `talk`, `start`, `step`, `report`
- `task_context`
  - 현재 task id, subgoal, return_required, user_anchor 등
- `inspection_target`
  - 예: `{"room":"living_room","object":"tv","attribute":"power_state"}`
- `task_result`
  - 예: `{"room":"거실","object":"tv","attribute":"power_state","value":"off","confidence":0.98}`

### 5.2 task type 구성

권장 데이터셋 패밀리는 다음 다섯 가지다.

| task_type | 목적 | 입력 특징 | target_text 형식 |
| --- | --- | --- | --- |
| `route` | TALK/NAV 분류 | 최근 대화 이력, 유저 발화 | `TALK` 또는 `NAV` |
| `talk` | 일반 대화 | 최근 대화 이력, 유저 발화 | 자연어 한국어 응답 |
| `nav_start` | 실행 승인 + 목표 분해 | 유저 요청, memory, user anchor | 첫 줄 자연어 + trace lines |
| `nav_step` | 현재 System2 step 유지 | 이미지, history, memory, events | `"<y>, <x>"`, `STOP`, `←`, `→`, `↓` |
| `talk_report` | 결과 보고 | task_result, dialog summary | 자연어 한국어 보고 |

### 5.3 반드시 추가해야 할 시나리오 유형

현재 seed dataset에는 없는 항목들이다.

- 상태 확인형
  - `tv가 꺼졌는지 확인`
  - `문이 열려 있는지 봐줘`
  - `컵이 비어 있는지 확인`
- 왕복 보고형
  - 목적지 이동 -> 상태 확인 -> 사용자 복귀 -> 결과 보고
- 대화 유지형
  - 인사, 짧은 잡담, 설명, 질문 응답
- 경계형
  - 말투는 질문처럼 들리지만 실제로는 embodied action이 필요한 요청
  - 예: `거실 좀 봐줄래?`, `저기 가서 확인해줘`
- ambiguity / fallback형
  - 불명확해서 `TALK`로 남아야 하는 예시
  - 예: `tv 어때?` without task result context

### 5.4 예시 레코드

#### route sample

```json
{
  "schema_version": "system2_talk_nav_lora_v1",
  "sample_id": "route_0001",
  "split": "train",
  "task_type": "route",
  "phase": "route",
  "instruction": "거실에 가서 tv가 꺼졌는지 확인해줘",
  "dialog_history": [
    {"role": "user", "text": "안녕"},
    {"role": "assistant", "text": "안녕하세요."}
  ],
  "events": {},
  "memory_context": null,
  "current_image": "",
  "history_images": [],
  "route_label": "NAV",
  "target_text": "NAV"
}
```

#### nav_start sample

```json
{
  "schema_version": "system2_talk_nav_lora_v1",
  "sample_id": "nav_start_0001",
  "split": "train",
  "task_type": "nav_start",
  "phase": "start",
  "instruction": "거실에 가서 tv가 꺼졌는지 확인해줘",
  "dialog_history": [],
  "events": {"task_state": "new"},
  "memory_context": {
    "instruction": "거실에 가서 tv가 꺼졌는지 확인해줘",
    "scratchpad": {
      "goal_summary": "Check whether the TV in the living room is off.",
      "checked_locations": [],
      "recent_hint": "User was last seen in the hallway.",
      "next_priority": "Go to the living room, verify TV power state, then return to the user."
    },
    "text_lines": [
      {"text": "Living room is connected to the hallway on the left.", "score": 4.5, "source_type": "place_memory"}
    ],
    "keyframes": [],
    "crop_path": ""
  },
  "task_context": {
    "return_required": true,
    "user_anchor": {"place_id": "hallway_user_last_seen", "room_id": "hallway"},
    "inspection_target": {"room": "living_room", "object": "tv", "attribute": "power_state"}
  },
  "target_text": "지금 확인하고 올게요.\n[목표] 거실에 가서 tv가 꺼졌는지 확인\n[NAV] 거실로 이동\n[NAV] tv가 꺼졌는지 확인\n[목표 완료]\n[목표] 유저에게 돌아가서 결과 보고\n[NAV] 유저에게 이동"
}
```

#### nav_step sample

`nav_step`은 현재 `system2_memory_lora` 레코드 구조를 거의 그대로 재사용한다.

```json
{
  "schema_version": "system2_talk_nav_lora_v1",
  "sample_id": "nav_step_0101",
  "split": "train",
  "task_type": "nav_step",
  "phase": "step",
  "instruction": "거실에 가서 tv가 꺼졌는지 확인해줘",
  "events": {"task_state": "active", "force_s2": false, "collision_risk": false},
  "memory_context": {"instruction": "거실에 가서 tv가 꺼졌는지 확인해줘"},
  "current_image": "images/train/nav_step_0101_current.jpg",
  "history_images": [{"frame_id": 101, "path": "images/train/nav_step_0101_history_1.jpg"}],
  "target_text": "233, 141"
}
```

#### talk_report sample

```json
{
  "schema_version": "system2_talk_nav_lora_v1",
  "sample_id": "talk_report_0001",
  "split": "train",
  "task_type": "talk_report",
  "phase": "report",
  "instruction": "거실에 가서 tv가 꺼졌는지 확인해줘",
  "dialog_history": [],
  "task_result": {
    "room": "거실",
    "object": "tv",
    "attribute": "power_state",
    "value": "off",
    "confidence": 0.98
  },
  "target_text": "거실에서 확인해본 결과, tv가 꺼져 있습니다."
}
```

## 6. 데이터 생성 전략

### 6.1 현재 코드베이스를 활용한 bootstrap

다음 자산을 그대로 활용한다.

- `artifacts/datasets/system2_memory_lora_seed`
  - `nav_step` smoke set의 시작점
- `System2Session` prompt/body builder
  - `nav_step` 입력 포맷의 source of truth
- `TaskRequest`, `RuntimeNotice`, `MemoryContextBundle`
  - `route`, `nav_start`, `talk_report` 레이블링의 source

### 6.2 권장 수집 경로

1. Isaac Sim scripted task rollout
   - `거실 가기`
   - `객체 상태 확인`
   - `유저 복귀`
   - `결과 보고`

2. 운영 로그 weak-label bootstrap
   - 현재 rule-based `ExecutionModeClassifier` 결과를 초기 라벨로 사용
   - 이후 사람이 경계 케이스를 재검수

3. synthetic paraphrase augmentation
   - 같은 intent를 다양한 말투로 증강
   - 예:
     - `거실에 가서 tv가 꺼졌는지 확인해줘`
     - `거실 TV 상태 좀 보고 와`
     - `TV가 꺼져 있는지 봐줘`

4. result report synthesis + human rewrite
   - task result 구조체에서 기계적으로 초안을 만들고, 자연스러운 한국어로 재작성

### 6.3 추천 데이터 비율

초기 joint SFT 기준 권장 비율:

- `route`: 10%
- `talk`: 25%
- `nav_start`: 15%
- `nav_step`: 40%
- `talk_report`: 10%

핵심은 `nav_step` 비중을 가장 크게 두되, `talk`가 충분히 많아야 기본 모드가 대화형으로 안정된다.

## 7. 학습 방법

### 7.1 단일 LoRA 유지

현재 `experiments/system2_memory_lora/system2_memory_lora.yaml`의 LoRA 설정을 그대로 출발점으로 쓰는 것이 가장 안전하다.

권장 유지 항목:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `multi_modal_projector`
- `lm_head`

즉, "TALK용 LoRA"와 "NAV용 LoRA"를 따로 두기보다, 하나의 joint LoRA를 만들고 프롬프트로 task를 구분한다.

### 7.2 권장 학습 순서

#### Stage A. NAV retention warm-start

- 목적: 현재 `System2Session`의 step contract를 먼저 보존
- 데이터: `nav_step` only
- 초기화:
  - 선택지 1: base VLM에서 시작
  - 선택지 2: 기존 `system2_memory_lora` 체크포인트에서 시작

권장:

- 기존 navigation LoRA가 있으면 그 위에서 시작
- 없으면 현재 seed + Isaac 확장셋으로 먼저 `nav_step` SFT를 수행

#### Stage B. Joint multitask SFT

- 데이터: `route + talk + nav_start + nav_step + talk_report`
- 방법:
  - 각 샘플에 `phase`와 `task_type`를 명시
  - assistant 정답 토큰에만 loss를 건다
  - batch는 task_type별 weighted sampler를 사용한다

#### Stage C. Hard-case replay

- 라우팅 실패 예시
- `NAV_START`에서 trace format이 깨진 예시
- `NAV_STEP`에서 문장형 출력이 섞인 예시
- `TALK_REPORT`에서 hallucination이 난 예시

위 실패 샘플만 따로 모아 짧게 replay fine-tuning 한다.

### 7.3 하이퍼파라미터 권장값

현재 config를 기준으로 시작하고, joint SFT에서는 learning rate만 약간 낮추는 것을 권장한다.

- `max_length`: 4096 유지
  - 대화 이력과 memory keyframe이 더 길어지면 6144까지 상향 검토
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 8
- `bf16`: true
- `gradient_checkpointing`: true
- `num_train_epochs`
  - Stage A: 1.0 ~ 2.0
  - Stage B: 1.0 ~ 1.5
- `learning_rate`
  - base model에서 바로 joint SFT: `2e-4`
  - 기존 nav LoRA에서 continue: `5e-5` ~ `1e-4`

### 7.4 중요한 학습 규칙

- `NAV_STEP`은 반드시 현재 파서와 호환되는 exact-format supervision을 쓴다.
- `NAV_START`는 자유로운 chain-of-thought를 학습시키지 말고, `[목표]`, `[NAV]`, `[목표 완료]`만 노출한다.
- `TALK_REPORT`는 오직 `task_result`에 있는 사실만 말하게 만든다.
- `ROUTE`는 무조건 한 토큰 정답만 학습시킨다.

## 8. 검증 항목

### 8.1 오프라인 metric

- `route` accuracy
- `talk` response quality
  - 짧은 human eval 또는 rubric scoring
- `nav_start` parse success rate
  - 첫 줄 자연어 + trace line format 유지율
- `nav_step` exact-match / valid-format rate
- `talk_report` groundedness
  - `task_result` 밖 사실 hallucination 비율

### 8.2 시나리오 단위 metric

필수 평가 시나리오:

1. `안녕`
   - route = `TALK`
   - talk output = 인사

2. `거실에 가서 tv가 꺼졌는지 확인해줘`
   - route = `NAV`
   - `NAV_START` 첫 줄 = 승인 발화
   - trace에 `이동 -> 확인 -> 유저 복귀`가 모두 포함
   - `NAV_STEP` 동안 invalid 문장 출력이 없어야 함
   - 완료 뒤 `TALK_REPORT`가 사실 기반으로 보고해야 함

### 8.3 실패 모드와 대응

- 문제: `NAV_STEP`에서 문장형 응답이 새어 나옴
  - 대응: `phase=step` 샘플 비중 증가, invalid output hard negative 추가

- 문제: `route`가 과하게 `NAV`를 선택
  - 대응: greeting / explanation / small talk `TALK` 샘플 대폭 보강

- 문제: 보고 문장에서 hallucination 발생
  - 대응: `task_result` 밖 정보는 금지하는 counter-example 추가

- 문제: check-and-return trace가 누락됨
  - 대응: `return_required=true`가 있는 `nav_start` 샘플을 별도 slice로 oversampling

## 9. 구현 메모

현재 코드 기준으로는 다음 방식이 가장 자연스럽다.

- `ExecutionModeClassifier`는 즉시 제거하지 말고 fallback으로 남긴다.
- 새 VLM router는 `TaskManager` 앞단 또는 그 안에서 시도한다.
- `System2Session`은 `NAV_STEP` 전용으로 유지한다.
- 새 `TalkNavSession` 또는 동등한 builder를 추가해
  - `ROUTE`
  - `TALK`
  - `NAV_START`
  - `TALK_REPORT`
  요청 본문을 구성한다.

중요한 점은, 현재 `System2Session`의 low-level navigation API를 깨지 않는 것이다. 이번 설계는 기존 `NAV step`을 보존하면서 그 위에 `route/talk/report`를 같은 백본과 같은 LoRA로 얹는 방향이다.
