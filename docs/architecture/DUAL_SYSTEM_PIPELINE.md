# Dual-System Pipeline

이 문서는 현재 `runtime.g1_bridge` 경로에서 사용되는 dual-system planning 흐름을 설명한다. 대상은 `scripts/powershell/run_pipeline.ps1`로 실행하는 G1 런타임이며, live smoke 진단 경로는 범위에 포함하지 않는다.

## 범위와 전제

- Canonical launcher: `scripts/powershell/run_pipeline.ps1`
- Runtime entry: `runtime.g1_bridge`
- Planner core: `runtime.planning_session`
- System 2: VLM 기반 목표 선택
- System 1: NavDP 기반 궤적 생성

현재 PowerShell 런처가 직접 허용하는 planner mode는 `interactive`, `pointgoal`뿐이다. 코드 레벨 `dual` 모드는 존재하지만, 일반 사용자는 보통 `interactive`에서 자연어 명령을 넣어 dual path를 활성화하게 된다.

## 주요 구성요소

- `scripts/powershell/run_pipeline.ps1`
  - Isaac Python과 런타임 인자를 정리하고 `runtime.g1_bridge`를 실행한다.
- `runtime.g1_bridge.NavDPCommandSource`
  - 프레임별 update 루프를 소유한다.
  - planning 결과를 저수준 명령으로 전달한다.
  - 필요 시 `Supervisor`로 perception/memory path도 병렬 유지한다.
- `runtime.planning_session.PlanningSession`
  - planner mode별 계획 로직의 중심이다.
  - point-goal, no-goal, dual planner를 초기화하고 캐시 상태를 관리한다.
- `services.dual_orchestrator.DualOrchestrator`
  - dual server 내부에서 System 2와 System 1 실행 여부를 결정한다.
  - goal/trajectory cache와 TTL 규칙을 관리한다.
- `runtime.subgoal_executor.SubgoalExecutor`
  - trajectory를 실제 `vx`, `vy`, `wz` 명령으로 변환한다.
- `runtime.supervisor.Supervisor`
  - 같은 프레임을 perception -> memory -> orchestration 경로에도 연결한다.

## Planner Mode 정리

### 1. `pointgoal`

가장 단순한 경로다.

- 사용자는 `--goal-x`, `--goal-y`를 직접 지정한다.
- `NavDPCommandSource`는 수동 `NAV_TO_POSE` 명령을 만든다.
- `PlanningSession`은 world goal을 planner 입력으로 변환한다.
- `AsyncPointGoalPlanner`가 NavDP를 호출한다.
- `SubgoalExecutor`가 반환 trajectory를 추종한다.

이 모드는 System 2를 사용하지 않는다.

### 2. `interactive`

현재 기본 모드다. 내부적으로 두 단계가 있다.

- `roaming`
- `task_active`

초기에는 `roaming` 상태로 시작해 no-goal planner를 사용한다. 사용자가 터미널에 자연어 지시를 입력하면 `task_active`로 전환되고 dual-system path가 활성화된다.

### 3. `dual` 코드 경로

코드상으로는 직접 dual mode를 시작할 수 있다.

- `PlanningSession.start_dual_task()`가 시작 instruction을 넣는다.
- startup 시 NavDP 서버와 dual server health를 모두 확인한다.
- 이후 프레임마다 dual planner 결과를 받아 바로 추종한다.

다만 현재 PowerShell 런처는 이 모드를 일반 옵션으로 노출하지 않는다.

## 하나의 프레임, 두 개의 소비자

G1 런타임의 핵심은 같은 Isaac 프레임을 planning path와 perception/memory path가 병렬로 소비한다는 점이다.

1. `PlanningSession.capture_observation()`가 RGB, depth, 카메라 pose, intrinsic을 읽는다.
2. `runtime.g1_bridge`는 같은 데이터를 `IsaacObservationBatch`로 감싼다.
3. 이 batch는 `Supervisor.process_frame()`로 들어가 perception/memory를 갱신한다.
4. 동시에 원본 observation은 planner path로 들어가 trajectory 계산에 사용된다.

중요한 점은 두 경로가 같은 프레임을 보지만 역할은 다르다는 것이다.

- perception/memory path
  - detector, tracker, projection, memory update, task context 유지
- planning path
  - System 1/2 또는 point-goal/no-goal planner로 실제 이동 trajectory 계산

즉, G1 pipeline은 “인지 아키텍처 관측 갱신”과 “실시간 locomotion 계획”을 같은 프레임으로 병렬 운영한다.

## Perception / Memory 병렬 경로

`runtime.g1_bridge`가 만든 `IsaacObservationBatch`는 다음 정보를 담는다.

- `FrameHeader`
- `rgb_image`
- `depth_image_m`
- `camera_intrinsic`
- `robot_pose_xyz`
- `robot_yaw_rad`
- planner overlay / capture metadata

이 batch는 `Supervisor.process_frame()`를 통해 다음 단계로 흐른다.

1. `PerceptionPipeline`이 detector/tracker/projection을 수행한다.
2. observation과 speaker event를 생성한다.
3. `TaskOrchestrator.on_observations()`와 `on_speaker_event()`가 호출된다.
4. `MemoryService.observe_objects()`가 spatial/temporal/episodic memory를 갱신한다.

다만 G1 pipeline에서는 locomotion 제어가 planner-managed command에 의해 우선되므로, `TaskOrchestrator`는 현재 프레임의 의미/기억 상태를 유지하는 병렬 소비자 역할에 더 가깝다.

## Planning / Locomotion 경로

### PointGoal 흐름

1. 런처가 goal 좌표를 전달한다.
2. `NavDPCommandSource`가 `NAV_TO_POSE` 수동 명령을 만든다.
3. `PlanningSession`이 robot-frame point-goal 입력으로 변환한다.
4. `AsyncPointGoalPlanner`가 NavDP를 호출한다.
5. `SubgoalExecutor`가 trajectory를 추종해 명령 벡터를 만든다.

### Interactive roaming 흐름

1. startup 시 `PlanningSession`이 roaming 상태를 활성화한다.
2. `AsyncNoGoalPlanner`가 no-goal trajectory를 계산한다.
3. `SubgoalExecutor`가 local roaming motion을 생성한다.
4. 사용자가 자연어 명령을 넣기 전까지 이 상태가 유지된다.

### Interactive task 흐름

1. 사용자가 터미널에 자연어 지시를 입력한다.
2. `runtime.g1_bridge`가 `submit_interactive_instruction()`을 호출한다.
3. `PlanningSession`이 pending instruction을 active task로 승격한다.
4. 이때 NavDP/dual service health와 `dual_reset(...)`이 수행된다.
5. 이후 프레임은 dual planner path를 타게 된다.

## System 2

System 2는 VLM 기반 목표 선택기다. 실제 구현 중심은 `services.dual_orchestrator.DualOrchestrator`와 `inference.vlm.System2Session`이다.

### 입력

- 현재 RGB frame
- 현재 instruction
- event flags
  - `force_s2`
  - `stuck`
  - `collision_risk`
- 이미지 크기와 history frame 정보

### 출력

System 2는 개념적으로 다음 정보를 반환한다.

- `pixel_x`
- `pixel_y`
- `stop`
- `reason`
- 필요 시 yaw 제어 정보와 재질의 필요 상태

### 현재 동작 특징

- goal 결과는 `GoalCache`로 캐시된다.
- 첫 결과가 즉시 `stop=true`여도 유효 trajectory가 아직 없으면 suppression이 걸릴 수 있다.
- 동일 목표는 `goal_version`을 유지할 수 있다.
- System 2는 `llm` 또는 `mock` 모드로 실행될 수 있다.

## System 1

System 1은 System 2가 선택한 pixel goal을 NavDP trajectory로 바꾸는 경로다.

### 입력

- RGB frame
- depth frame
- `pixel_goal`
- `sensor_meta`
- `cam_pos`
- `cam_quat_wxyz`

### 실행

- `PlanningSession`은 `AsyncDualPlanner`를 통해 dual server를 호출한다.
- dual server 내부에서는 `DualOrchestrator.step()`이 System 1/2 실행 여부를 정한다.
- System 1은 NavDP 경로를 통해 world trajectory를 계산한다.

### 출력

- `trajectory_world`
- `goal_version`
- `traj_version`
- `stale_sec`
- debug state

### 현재 동작 특징

- trajectory는 `TrajectoryCache`로 캐시된다.
- goal이 바뀐 뒤 늦게 도착한 stale trajectory는 버려질 수 있다.
- `traj_ttl_sec`, `traj_max_stale_sec`를 넘어가면 System 2 재질의가 강제될 수 있다.

## Dual Task 활성 상태의 End-to-End 흐름

1. Isaac Sim이 RGB, depth, 카메라 pose, 로봇 pose를 생성한다.
2. `PlanningSession.capture_observation()`가 한 번 observation을 만든다.
3. 같은 프레임이 `Supervisor`로 가서 perception/memory를 갱신한다.
4. 동시에 observation이 `AsyncDualPlanner` 쪽으로 전달된다.
5. dual server는 현재 cache/TTL/event 상태를 보고 다음 중 하나를 선택한다.
   - System 2만 실행
   - System 1만 실행
   - 둘 다 실행
   - 둘 다 건너뛰고 cache 재사용
6. 최신 pixel goal과 trajectory가 반환된다.
7. `PlanningSession`은 최신 버전과 stale 여부를 검증한다.
8. `SubgoalExecutor`가 trajectory를 추종한다.
9. 최종적으로 `vx`, `vy`, `wz` 명령 벡터가 생성된다.

## 운영상 구분해야 할 상태

현재 G1 runtime에서 아래 상태는 서로 다르다.

- perception pipeline이 정상 동작하는 상태
- no-goal roaming 상태
- 자연어 지시 이후 dual task execution 상태

같은 프레임을 공유하더라도 이것들을 동일한 상태로 보면 안 된다.

## 현재 한계

- `run_pipeline.ps1`는 `planner-mode=dual`을 직접 노출하지 않는다.
- dual path는 NavDP server와 dual server가 모두 살아 있어야 한다.
- `Supervisor`는 perception/memory를 유지하지만, G1 runtime의 locomotion 제어권은 planner path가 가진다.
- viewer 연동, remote service health, Isaac sensor bootstrap 상태가 실제 dual runtime의 안정성에 직접 영향을 준다.
