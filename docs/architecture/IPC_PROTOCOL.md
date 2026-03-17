# IPC Protocol

AURA의 기본 런타임은 Flask/HTTP 중심이 아니라 message bus 기반 direct IPC다. 현재 저장소에서 고수준 태스크, observation header, action command, 상태 보고는 버스를 통해 오가고, 큰 RGB/depth payload는 필요하면 shared memory로 분리된다.

## 기본 원칙

- 기본 제어 경로는 `MessageBus` 인터페이스 위에 구현된다.
- 단일 프로세스에서는 `InprocBus`를 사용한다.
- 다중 프로세스에서는 `ZmqBus`를 사용한다.
- 큰 frame payload는 `SharedMemoryRing`으로 넘기고, 버스에는 참조만 싣는다.
- legacy HTTP wrapper는 남아 있지만, 현재 권장 아키텍처의 중심은 아니다.

## 주요 컴포넌트

- `ipc.base.MessageBus`
  - 런타임이 의존하는 공통 publish/poll 인터페이스
- `ipc.inproc_bus.InprocBus`
  - 단일 프로세스 loopback/debug용
- `ipc.zmq_bus.ZmqBus`
  - bridge/agent 분리용
  - control plane, telemetry plane을 분리
- `ipc.shm_ring.SharedMemoryRing`
  - RGB/depth blob 저장
- `adapters.sensors.isaac_bridge_adapter.IsaacBridgeAdapter`
  - observation/task/status/health 토픽 계약을 감싸는 어댑터

## 현재 토픽 계약

`IsaacBridgeAdapterConfig` 기준 토픽은 다음과 같다.

- `isaac.observation`
- `isaac.task`
- `isaac.command`
- `isaac.status`
- `isaac.notice`
- `isaac.capability`
- `isaac.health`

실제 message type 매핑은 다음과 같다.

- `isaac.observation` -> `FrameHeader`
- `isaac.task` -> `TaskRequest`
- `isaac.command` -> `ActionCommand`
- `isaac.status` -> `ActionStatus`
- `isaac.notice` -> `RuntimeNotice`
- `isaac.capability` -> `CapabilityReport`
- `isaac.health` -> `HealthPing`

## 메시지 타입 요약

### `FrameHeader`

frame 전체를 담는 구조가 아니라, frame payload를 재구성하기 위한 header다.

포함 정보:

- `frame_id`
- `timestamp_ns`
- `source`
- image width/height
- camera pose / robot pose / sim time
- `metadata`

중요한 점은 RGB/depth 자체가 항상 header 본문에 들어가는 것이 아니라는 점이다. 현재 구현은 다음 두 방식 중 하나를 사용한다.

- inline hex payload
- shared-memory ref

그리고 `camera_intrinsic`, `capture_report`, `speaker_events` 같은 부가 정보는 `FrameHeader.metadata`에 직렬화된다.

### `TaskRequest`

자연어 명령과 target metadata를 memory/orchestration 쪽으로 전달한다.

주요 필드:

- `command_text`
- `task_id`
- `intent`
- `target_json`
- `speaker_id`

### `ActionCommand`

상위 태스크가 실제 행동 단위로 변환된 결과다.

현재 action type은 다음을 포함한다.

- `STOP`
- `LOOK_AT`
- `FOLLOW_TARGET`
- `FOLLOW_PERSON`
- `NAV_TO_PLACE`
- `NAV_TO_POSE`
- `LOCAL_SEARCH`

### `ActionStatus`

저수준 실행 상태를 상위 레이어로 올린다.

- `idle`
- `running`
- `succeeded`
- `failed`
- `stale`

### `CapabilityReport`, `RuntimeNotice`, `HealthPing`

이 세 타입은 데이터 평면보다 운영 진단에 가깝다.

- detector/runtime backend 상태 보고
- fallback/경고/오류 공지
- bridge/agent 생존 신호 및 snapshot 요약

## 전송 평면

`ZmqBus`는 현재 control plane과 telemetry plane을 분리한다.

### Control plane

주로 작은 제어/상태성 메시지를 다룬다.

- `isaac.task`
- `isaac.command`
- `isaac.notice`
- `isaac.capability`

브리지 역할은 `ROUTER`, 에이전트 역할은 `DEALER`를 사용한다. 일부 control topic은 retained replay가 있어 늦게 붙은 peer가 최근 상태를 다시 받을 수 있다.

### Telemetry plane

프레임 및 실행 상태 스트림을 다룬다.

- `isaac.observation`
- `isaac.status`
- `isaac.health`

브리지 역할은 `PUB`, 에이전트 역할은 `SUB`를 사용한다.

## Observation Batch 재구성 방식

실제 런타임은 `FrameHeader`만으로 끝나지 않는다. 현재 구조는 `IsaacBridgeAdapter`가 batch를 publish/reconstruct하는 방식으로 동작한다.

1. 브리지가 `IsaacObservationBatch`를 만든다.
2. RGB/depth가 있으면 inline 또는 SHM ref로 변환한다.
3. `FrameHeader`를 `isaac.observation` 토픽에 발행한다.
4. 수신 측은 `reconstruct_batch(...)`로 RGB/depth/intrinsic/speaker event를 복원한다.
5. `Supervisor`가 복원된 batch를 perception/memory로 흘린다.

이 구조 덕분에 버스는 작은 header 중심으로 유지하고, 큰 이미지 payload는 필요 시 별도 메모리 공간으로 분리할 수 있다.

## 대표적인 런타임 흐름

### Deprecated Local Stack

- `InprocBus`
- SHM 없이 같은 프로세스에서 batch 소비
- `Supervisor`가 바로 perception -> memory -> orchestration 수행

### Two-process Bridge + Memory Agent

1. Isaac Bridge가 frame/task/status를 발행한다.
2. Memory Agent가 `FrameHeader`를 읽고 batch를 재구성한다.
3. `Supervisor`가 메모리와 오케스트레이션을 수행한다.
4. Memory Agent가 `ActionCommand`를 발행한다.
5. Bridge가 low-level 실행과 `ActionStatus` 보고를 담당한다.

### `g1_view`

G1 pipeline의 viewer 모드에서는 같은 관측이 viewer overlay와 함께 외부 소비자에게 공개된다. 즉, IPC는 deprecated local stack과 supporting memory agent뿐 아니라 G1 시각화 경로와도 공유되는 계약이다.

## Deprecated Live Smoke와의 정합성

Live Smoke는 메인 runtime은 아니며 decommission 대상이지만, 가능한 한 같은 observation/message 형태를 재사용한다.

- perception ingress 확인
- memory ingress 확인
- diagnostics JSON 생성

즉, live smoke는 별도 프로토콜을 만드는 대신 운영 진단을 위해 현재 IPC 계약에 최대한 맞춰진 상태를 유지한다.

## 현재 한계

- targeted routing보다 shared-topic 기반 fan-out에 가깝다.
- `editor_assisted`, `extension_mode`는 여전히 in-editor 실행이 전제된다.
- 관측 payload는 버스 메시지 하나만으로 완결되지 않고 metadata + SHM 복원이 필요할 수 있다.
