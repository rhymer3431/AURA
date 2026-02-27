# Isaac Sim ROS2 Integration for decoupled_wbc

## 1. 구현 요약 (선택 아키텍처와 이유)

선택안은 **A안: decoupled_wbc에 Isaac ROS2 adapter 계층 추가**입니다.

- 이유 1: `run_g1_control_loop.py`/`G1Env`/`state_processor.py`/`command_sender.py` 기반 실배포 경로를 유지한 채, `SIMULATOR=isaacsim` 조건에서만 신규 경로를 활성화할 수 있습니다.
- 이유 2: 기존 ByteMultiArray+msgpack 스키마(`ControlPolicy/upper_body_pose`, `G1Env/env_state_act`)와 호환성을 유지하면서 Isaac 표준 토픽(`/joint_states`, `/tf`, `/clock`, `/imu`)을 흡수할 수 있습니다.
- 이유 3: gear_sonic_deploy ROS2 핸들러와의 상호운용(동일 스키마 키)을 유지합니다.

비선택안(B안) 요약:
- `gear_sonic_deploy --input-type ros2 --output-type ros2`는 이미 강한 ROS2 경로가 있으나, decoupled_wbc 메인 루프를 기본 경로로 유지한다는 요구에 비해 런타임/빌드 의존성이 증가합니다.
- 따라서 B안은 **호환/대체 경로**로 유지하고, 기본 통합 경로는 A안으로 구성했습니다.

---

## 2. 변경 파일 목록

### 신규 파일
- `decoupled_wbc/control/envs/g1/utils/isaac_ros_interface.py`
- `decoupled_wbc/control/utils/isaac_ros_adapter.py`
- `decoupled_wbc/control/main/teleop/run_isaac_ros2_adapter.py`
- `tools/isaac/load_g1_usd_ros2.py`

### 수정 파일
- `decoupled_wbc/control/envs/g1/g1_body.py`
- `decoupled_wbc/control/envs/g1/g1_hand.py`
- `decoupled_wbc/control/envs/g1/g1_env.py`
- `decoupled_wbc/control/envs/g1/sim/simulator_factory.py`
- `decoupled_wbc/control/main/constants.py`
- `decoupled_wbc/control/main/teleop/configs/configs.py`
- `decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py`

---

## 3. 인터페이스 감사 (코드 추출)

### 3.1 내부/기존 인터페이스

| Topic/Service | 방향 | 메시지 타입 | 목표 Hz | 페이로드 키(핵심) | 코드 근거 |
|---|---|---|---|---|---|
| `ControlPolicy/upper_body_pose` | Teleop Pub -> Control Sub | `std_msgs/ByteMultiArray` + msgpack(dict) | `teleop_frequency` (기본 20Hz) | `navigate_cmd`, `wrist_pose`, `left_wrist_after_ik`, `right_wrist_after_ik`, `head_after_ik`, `left_hand_joint`, `right_hand_joint`, `base_height_command`, `toggle_policy_action`, `locomotion_mode`, `ros_timestamp`, `target_upper_body_pose`, `target_time` | `run_teleop_policy_loop.py`, `run_g1_control_loop.py`, `ros2_input_handler.hpp` |
| `G1Env/env_state_act` | Control Pub -> Exporter Sub | `std_msgs/ByteMultiArray` + msgpack(dict) | `control_frequency` (기본 50Hz) | decoupled_wbc 경로: `q,dq,ddq,tau_est,floating_base_pose,floating_base_vel,floating_base_acc,wrist_pose,torso_quat,torso_ang_vel,action,action.eef,base_height_command,navigate_command,timestamps` | `run_g1_control_loop.py`, `run_g1_data_exporter.py` |
| `WBCPolicy/robot_config` | decoupled_wbc: Service Server/Client, gear: Topic Pub | decoupled_wbc=`std_srvs/Trigger`(base64(msgpack)); gear=`std_msgs/ByteMultiArray`(msgpack) | 1회/요청 시 | WBC/로봇 설정 dict | `run_g1_control_loop.py`, `run_g1_data_exporter.py`, `ros2_output_handler.hpp` |

### 3.2 Isaac 표준 토픽 + 신규 Adapter 인터페이스

| Topic | 방향 (Adapter 기준) | 메시지 타입 | 목표 Hz | 페이로드 키/필드 | 코드 근거 |
|---|---|---|---|---|---|
| `/joint_states` | Sub | `sensor_msgs/JointState` | Isaac physics/control tick (권장 >=50Hz) | `name, position, velocity, effort` | `isaac_ros_adapter.py` |
| `/tf` | Sub | `tf2_msgs/TFMessage` | Isaac publish tick | `odom_frame -> base_frame` transform 사용 | `isaac_ros_adapter.py` |
| `/clock` | Sub | `rosgraph_msgs/Clock` | Isaac sim clock tick | `clock.sec`, `clock.nanosec` | `isaac_ros_adapter.py` |
| `/imu` | Sub | `sensor_msgs/Imu` | Isaac IMU sensor tick | `orientation`, `angular_velocity`, `linear_acceleration` | `isaac_ros_adapter.py` |
| `G1Env/isaac_state` | Pub | `std_msgs/ByteMultiArray` + msgpack(dict) | `state_publish_hz` (기본 100Hz) | `floating_base_pose`, `floating_base_vel`, `floating_base_acc`, `body_q/dq/ddq/tau_est`, `left_hand_*`, `right_hand_*`, `torso_quat`, `torso_ang_vel`, `ros_timestamp`, `foot_contact` | `isaac_ros_adapter.py` |
| `G1Env/isaac_joint_command` | Sub(bridge) / Pub(control sender) | `std_msgs/ByteMultiArray` + msgpack(dict) | control loop tick (기본 50Hz) | `body_q`, `body_dq`, `body_tau`, `left_hand_q`, `right_hand_q`, `ros_timestamp`, `source` | `isaac_ros_interface.py`, `isaac_ros_adapter.py` |
| `/isaac/joint_command` | Pub | `sensor_msgs/JointState` | control loop tick (기본 50Hz) | `name`, `position`, `velocity`, `effort` | `isaac_ros_adapter.py` |
| `foot_contact`(옵션) | 현재 내부 payload key만 유지 | msgpack field | n/a | 현재 빈 배열(`[]`)로 채움 | `isaac_ros_adapter.py` |

---

## 4. Isaac <-> 내부 매핑표

### 4.1 단위/축/좌표
- Joint position: rad
- Joint velocity: rad/s
- Joint effort/torque: Nm
- Base linear position: m
- Base linear velocity: m/s
- Base angular velocity: rad/s
- Quaternion: 내부는 `w,x,y,z` 순서 (ROS `x,y,z,w` -> 변환)

### 4.2 조인트 순서 (29 DoF body)

| idx | joint_name |
|---:|---|
| 0 | left_hip_pitch_joint |
| 1 | left_hip_roll_joint |
| 2 | left_hip_yaw_joint |
| 3 | left_knee_joint |
| 4 | left_ankle_pitch_joint |
| 5 | left_ankle_roll_joint |
| 6 | right_hip_pitch_joint |
| 7 | right_hip_roll_joint |
| 8 | right_hip_yaw_joint |
| 9 | right_knee_joint |
| 10 | right_ankle_pitch_joint |
| 11 | right_ankle_roll_joint |
| 12 | waist_yaw_joint |
| 13 | waist_roll_joint |
| 14 | waist_pitch_joint |
| 15 | left_shoulder_pitch_joint |
| 16 | left_shoulder_roll_joint |
| 17 | left_shoulder_yaw_joint |
| 18 | left_elbow_joint |
| 19 | left_wrist_roll_joint |
| 20 | left_wrist_pitch_joint |
| 21 | left_wrist_yaw_joint |
| 22 | right_shoulder_pitch_joint |
| 23 | right_shoulder_roll_joint |
| 24 | right_shoulder_yaw_joint |
| 25 | right_elbow_joint |
| 26 | right_wrist_roll_joint |
| 27 | right_wrist_pitch_joint |
| 28 | right_wrist_yaw_joint |

### 4.3 손 조인트 순서 (7 + 7)

| hand | idx | joint_name |
|---|---:|---|
| left | 0 | left_hand_thumb_0_joint |
| left | 1 | left_hand_thumb_1_joint |
| left | 2 | left_hand_thumb_2_joint |
| left | 3 | left_hand_index_0_joint |
| left | 4 | left_hand_index_1_joint |
| left | 5 | left_hand_middle_0_joint |
| left | 6 | left_hand_middle_1_joint |
| right | 0 | right_hand_thumb_0_joint |
| right | 1 | right_hand_thumb_1_joint |
| right | 2 | right_hand_thumb_2_joint |
| right | 3 | right_hand_index_0_joint |
| right | 4 | right_hand_index_1_joint |
| right | 5 | right_hand_middle_0_joint |
| right | 6 | right_hand_middle_1_joint |

### 4.4 ControlGoal 최소 키 호환

`run_teleop_policy_loop.py`에서 아래 키를 기본 보강하여 gear_sonic_deploy ROS2 입력 파서와 호환:
- `navigate_cmd`
- `wrist_pose`
- `left_wrist_after_ik`
- `right_wrist_after_ik`
- `head_after_ik`
- `left_hand_joint`
- `right_hand_joint`
- `base_height_command`
- `toggle_policy_action`
- `locomotion_mode`
- `ros_timestamp`

---

## 5. 실행 명령어 (복붙)

> Assumption: ROS2 환경과 Isaac Sim 설치 경로가 준비되어 있고, ROS_DOMAIN_ID는 동일(`0`)로 사용.

### 5.1 공통 환경 (PowerShell)

```powershell
Set-Location c:\Users\mango\project\sonic\GR00T-WholeBodyControl
$env:ROS_DOMAIN_ID="0"
$env:RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
```

### 5.2 Isaac Sim 실행 + ROS2 Bridge 활성화 + USD 로드

```powershell
$env:ISAAC_SIM_ROOT="C:\isaacsim"
& "$env:ISAAC_SIM_ROOT\python.bat" tools/isaac/load_g1_usd_ros2.py `
  --usd-path "c:\Users\mango\project\sonic\GR00T-WholeBodyControl\gear_sonic\data\robot_model\model_data\g1\g1_29dof_with_hand\g1_29dof_with_hand.usd"
```

브리지 시작 에러(`ROS2 Bridge startup failed`)가 나면:
```powershell
& "$env:ISAAC_SIM_ROOT\python.bat" tools/isaac/load_g1_usd_ros2.py `
  --usd-path "c:\Users\mango\project\sonic\GR00T-WholeBodyControl\gear_sonic\data\robot_model\model_data\g1\g1_29dof_with_hand\g1_29dof_with_hand.usd" `
  --ros2-bridge-ext auto `
  --rmw-implementation rmw_fastrtps_cpp
```
위 실행은 환경 변수 스냅샷과 확장 활성화 시도(`isaacsim.ros2.bridge`, `omni.isaac.ros2_bridge`)를 모두 출력합니다.

### 5.3 Adapter 노드 실행

```powershell
python decoupled_wbc/control/main/teleop/run_isaac_ros2_adapter.py `
  --run-mode both `
  --with-hands True `
  --state-publish-hz 100 `
  --joint-states-topic /joint_states `
  --tf-topic /tf `
  --clock-topic /clock `
  --imu-topic /imu `
  --internal-state-topic G1Env/isaac_state `
  --internal-command-topic G1Env/isaac_joint_command `
  --isaac-command-topic /isaac/joint_command
```

### 5.4 decoupled_wbc control loop 실행

```powershell
python decoupled_wbc/control/main/teleop/run_g1_control_loop.py `
  --interface sim `
  --simulator isaacsim `
  --control-frequency 50 `
  --isaac-internal-state-topic G1Env/isaac_state `
  --isaac-internal-command-topic G1Env/isaac_joint_command
```

### 5.5 Teleop / Data Exporter (선택)

```powershell
python decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py `
  --interface sim `
  --simulator isaacsim `
  --teleop-frequency 20
```

```powershell
python decoupled_wbc/control/main/teleop/run_g1_data_exporter.py `
  --interface sim `
  --simulator isaacsim
```

### 5.6 gear_sonic_deploy 경로 (선택)

```powershell
Set-Location c:\Users\mango\project\sonic\GR00T-WholeBodyControl\gear_sonic_deploy
$env:HAS_ROS2="1"

# 환경에 맞는 기존 빌드 커맨드 사용(예: just build 또는 CMake 빌드)
just build

# ROS2 I/O 사용 예시
just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ `
  --planner-file planner/target_vel/V2/planner_sonic.onnx `
  --input-type ros2 `
  --output-type ros2
```

---

## 6. 검증 결과 / 기준

### 6.1 이번 변경에서 수행한 검증
- Python 문법/컴파일 검증 통과:
  - `python -m py_compile`로 신규/수정 파일 컴파일 확인 완료.

### 6.2 런타임 검증 명령 (현장/시뮬 환경에서 실행)

```powershell
ros2 topic hz /clock
ros2 topic hz /joint_states
ros2 topic hz ControlPolicy/upper_body_pose
ros2 topic hz G1Env/env_state_act
```

```powershell
ros2 topic echo --once /joint_states
ros2 topic echo --once /imu
ros2 topic echo --once /tf
ros2 topic echo --once ControlPolicy/upper_body_pose
ros2 topic echo --once G1Env/env_state_act
ros2 topic echo --once G1Env/isaac_state
ros2 topic echo --once /isaac/joint_command
```

```powershell
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo odom base_link
```

관절 추종 확인:
- Teleop command 입력 시 `/isaac/joint_command`의 `position` 값이 변하는지 확인
- Isaac 측에서 해당 토픽을 실제 articulation command에 연결했는지 확인

---

## 7. 실패 시 디버깅 체크리스트

- 토픽 누락:
  - `/joint_states`, `/tf`, `/clock`, `/imu`가 실제 publish 중인지 확인
  - `G1Env/isaac_state`, `G1Env/isaac_joint_command`, `/isaac/joint_command`가 연결되어 있는지 확인
- QoS:
  - `/clock`/`/tf`/sensor 토픽 QoS 불일치 시 adapter에서 데이터 유실 가능
- Time sync:
  - `/clock` 미수신 시 wall clock fallback 사용됨. sim time 정합 필요 시 `/clock` 필수
- 조인트 매핑 불일치:
  - Isaac의 `JointState.name`이 문서의 29+14 이름과 다르면 값이 0으로 채워짐
- 프레임 불일치:
  - `/tf`에서 `odom -> base_link`를 못 찾으면 base pose가 기본값으로 유지됨
- 핸드 데이터:
  - teleop 장치가 hand angle을 제공하지 않으면 `left/right_hand_joint`는 0 기본값
- 명령 토픽 반영 실패:
  - Isaac side graph(Articulation Controller)에서 `/isaac/joint_command` 수신 연결 확인

---

## 8. Assumptions

- Isaac Sim ROS2 Bridge extension 이름은 `isaacsim.ros2.bridge`.
- Isaac stage의 joint 이름이 본 문서의 G1 29+14 명칭과 동일.
- `WBCPolicy/robot_config`는 decoupled_wbc에서는 서비스(`Trigger`) 기반이 기본이며, gear_sonic_deploy ROS2 출력은 topic 기반으로 별도 호환됨.
