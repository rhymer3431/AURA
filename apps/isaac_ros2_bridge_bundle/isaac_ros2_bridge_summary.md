# Isaac Sim ROS2 Bridge 정리본

## 포함 파일
- decoupled_wbc/control/utils/isaac_ros_adapter.py
- decoupled_wbc/control/main/teleop/run_isaac_ros2_adapter.py
- decoupled_wbc/control/envs/g1/utils/isaac_ros_interface.py
- decoupled_wbc/control/envs/g1/g1_body.py
- decoupled_wbc/control/envs/g1/g1_hand.py
- decoupled_wbc/control/envs/g1/sim/simulator_factory.py
- decoupled_wbc/control/envs/g1/g1_env.py
- decoupled_wbc/control/main/constants.py
- decoupled_wbc/control/main/teleop/configs/configs.py
- decoupled_wbc/control/main/teleop/run_g1_control_loop.py
- decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py
- tools/isaac/load_g1_usd_ros2.py
- docs/isaac_ros2_decoupled_wbc_integration.md

## 컨텍스트 요약
1. Isaac 쪽(시뮬레이터)에서 `/joint_states`, `/tf`, `/clock`, `/imu`를 받아 `G1Env/isaac_state`로 재패킹한다.
   - 구현: `decoupled_wbc/control/utils/isaac_ros_adapter.py`
   - 클래스: `IsaacToInternalStateBridge`

2. decoupled_wbc 내부 제어기에서 `G1Env/isaac_joint_command`를 받아서 `/isaac/joint_command`로 변환해 전달한다.
   - 구현: `decoupled_wbc/control/utils/isaac_ros_adapter.py`
   - 클래스: `InternalCommandToIsaacBridge`

3. Adapter 노드는 별도 엔트리 포인트로 기동한다.
   - 구현: `decoupled_wbc/control/main/teleop/run_isaac_ros2_adapter.py`
   - `--run-mode both|state|command`

4. `G1Body`, `G1ThreeFingerHand`가 `SIMULATOR=isaacsim`에서 Isaac 전용 Processor/Sender를 사용한다.
   - 구현: `decoupled_wbc/control/envs/g1/g1_body.py`, `decoupled_wbc/control/envs/g1/g1_hand.py`
   - 대상 클래스: `IsaacBodyStateProcessor`, `IsaacHandStateProcessor`, `IsaacBodyCommandSender`, `IsaacHandCommandSender`
   - 경로 구현: `decoupled_wbc/control/envs/g1/utils/isaac_ros_interface.py`

5. Simulator 팩토리에서 `SIMULATOR=isaacsim` 분기 시 Isaac는 내부 물리 시뮬레이션을 생성하지 않고 외부 ROS2 브릿지 경로로 동작한다.
   - 구현: `decoupled_wbc/control/envs/g1/sim/simulator_factory.py`

6. 제어 루프(`run_g1_control_loop.py`)는 위 설정을 받아 환경을 구성하므로 `--simulator isaacsim`이 연동 스위치 역할을 한다.
   - 구현: `decoupled_wbc/control/main/teleop/run_g1_control_loop.py`

7. 설정은 `decoupled_wbc/control/main/constants.py`의 Topic 상수와
   `decoupled_wbc/control/main/teleop/configs/configs.py`의 `simulator`, `isaac_internal_state_topic`, `isaac_internal_command_topic`로 일괄 주입된다.

8. 실행 순서 요약(예시)
   - `tools/isaac/load_g1_usd_ros2.py`로 Isaac Sim + ROS2 bridge 기동
   - `run_isaac_ros2_adapter.py` 실행
   - `run_g1_control_loop.py --simulator isaacsim` 실행

## 최근 분석 추가(현재 워크플로우와의 정합성)
- 질문한 `start_all.ps1` 흐름은 기본적으로 `agent_runtime + run_headless + (옵션)` 조합으로, decoupled_wbc 어댑터가 쓰는 토픽 세트와는 다르게 동작한다.
  - `agent_runtime` 경로: 기본적으로 `/g1/cmd/joint_commands`, `/g1/joint_states` 축으로 구성됨.
  - `decoupled_wbc` 경로: `/joint_states`, `/tf`, `/clock`, `/imu` -> `G1Env/isaac_state`, `G1Env/isaac_joint_command` -> `/isaac/joint_command`.
- 때문에 bundle를 그대로 붙이면 핵심 충돌 포인트는 토픽 네임/포맷 매핑이다.
  - `G1Env/isaac_state` 생성기: `IsaacToInternalStateBridge`(`sensor_msgs/JointState` + `/tf` + `/imu` 입력 집계).
  - 내부 명령 소비기: `IsaacBodyCommandSender`/`IsaacHandCommandSender`(`ByteMultiArray`/msgpack, `G1Env/isaac_joint_command`).
  - Isaac 출력 브리지: `InternalCommandToIsaacBridge`(`G1Env/isaac_joint_command` -> `/isaac/joint_command`, `sensor_msgs/JointState`).
- 정리된 연결 아이디어
  - 1) `run_headless.py`의 네임스페이스/토픽을 decoupled_wbc 기준으로 맞추는 직접 수정(가장 직관적)
  - 2) 중간에 간단한 ROS2 토픽 리맵/번역 노드 추가 (`/g1/...` <-> `/isaac/...`, `g1` namespace ↔ 최상위)
  - 3) 또는 `isaac_ros2_bridge_bundle` 상수만 `DEC_...`에서 기존 런타임 토픽으로 바꾸는 역방향 매핑(빠르지만 기존 설계 대비 추적 난이도 상승)

## SONIC/정책-모션 루프 질의 정리
- 사용자가 정리한 의도대로, gear sonic은 정책 서버(또는 WBC policy)로부터 나온 결정(몸/손 명령)을 받아 액추에이터 커맨드로 변환되어 실행된다.
- decoupled_wbc 구조에서는 `WBC`가 `env_state_act`에서 action을 생성하고, 내부 커맨드 토픽(`G1Env/isaac_joint_command`)을 통해 실행점으로 전달된다.
- 따라서 "몸 움직임을 Sonic policy가 제어하고, Gear/SIM이 정책에 따라 액션 수행"은 동작 방향이 성립한다. 다만 현재 `start_all.ps1` 기본 런타임은 반드시 decoupled_wbc 파이프라인으로 가는지 추가 확인이 필요하다.

## 참고
- 상세 실행 예시는 `docs/isaac_ros2_decoupled_wbc_integration.md` 참고
