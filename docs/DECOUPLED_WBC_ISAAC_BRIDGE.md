# decoupled_wbc + Isaac Sim ROS2 Bridge + GEAR-SONIC

이 프로젝트에서는 `apps/isaac_ros2_bridge_bundle`에 들어있는 Isaac 연동 패치를 공식 `decoupled_wbc` 코드 트리에 동기화한 뒤 실행합니다.

## 1) 패치 동기화

```powershell
cd C:\Users\mango\project\isaac-aura
.\scripts\sync_decoupled_wbc_isaac_bridge_bundle.ps1
```

기본 대상 경로:
- `C:\Users\mango\project\isaac-aura\apps\decoupled_wbc_workspace`

다른 경로를 쓰려면:

```powershell
.\scripts\sync_decoupled_wbc_isaac_bridge_bundle.ps1 -DecoupledWbcRoot "D:\GR00T-WholeBodyControl"
```

## 2) 통합 실행 (Isaac + Adapter + decoupled control loop)

기본 USD:
- `apps\isaac_ros2_bridge_bundle\robot_model\model_data\g1\g1_29dof_with_hand\g1_29dof_with_hand.usd`

```powershell
cd C:\Users\mango\project\isaac-aura
.\scripts\start_decoupled_wbc_isaac_bridge.ps1
```

이 명령은 다음을 순서대로 띄웁니다.
- `apps/isaacsim_runner/run_headless.py` (ROS2 bridge 토픽 포함)
- `run_isaac_ros2_adapter.py`
- `run_g1_control_loop.py --simulator isaacsim`

## 3) 옵션

Teleop 루프도 함께 실행:

```powershell
.\scripts\start_decoupled_wbc_isaac_bridge.ps1 -StartTeleop
```

GEAR-SONIC 서버도 함께 실행:

```powershell
.\scripts\start_decoupled_wbc_isaac_bridge.ps1 -StartSonicServer
```

GUI Isaac 실행:

```powershell
.\scripts\start_decoupled_wbc_isaac_bridge.ps1 -IsaacGui
```

다른 `decoupled_wbc` 루트를 사용:

```powershell
.\scripts\start_decoupled_wbc_isaac_bridge.ps1 -DecoupledWbcRoot "D:\GR00T-WholeBodyControl"
```

## 4) 기본 토픽 연결

- Isaac publish: `/<namespace>/joint_states`, `/<namespace>/imu`, `/tf`, `/clock`
- Adapter internal: `G1Env/isaac_state`, `G1Env/isaac_joint_command`
- Isaac command: `/<namespace>/cmd/joint_commands`

기본 namespace는 `g1`입니다.

## 5) 키보드 Kinematic Planner 포함 파이프라인 실행

`decoupled_wbc bridge` 경로에서 키보드 planner(`deploy.sh --input-type keyboard`)를 함께 올리려면 아래 스크립트를 사용합니다.

```powershell
cd C:\Users\mango\project\isaac-aura
.\scripts\start_decoupled_wbc_keyboard_planner.ps1
```

GUI로 Isaac 창까지 확인하려면:

```powershell
cd C:\Users\mango\project\isaac-aura
.\scripts\start_decoupled_wbc_keyboard_planner_gui.ps1
```

기본 동작:
- `start_decoupled_wbc_isaac_bridge.ps1`를 먼저 실행 (Isaac + adapter + WBC control loop)
- 이후 `apps/decoupled_wbc_workspace/gear_sonic_deploy/deploy.sh --input-type keyboard --output-type ros2 sim`를 실행
- 둘 중 하나가 종료되면 나머지 프로세스 정리

주요 옵션:

```powershell
# 특정 WSL distro에서 planner 실행
.\scripts\start_decoupled_wbc_keyboard_planner.ps1 -WslDistro Ubuntu-22.04

# planner 자동 승인 비활성화 (deploy.sh의 Proceed 프롬프트 수동 입력)
.\scripts\start_decoupled_wbc_keyboard_planner.ps1 -KeyboardPlannerAutoApprove:$false

# planner 추가 인자 전달
.\scripts\start_decoupled_wbc_keyboard_planner.ps1 -KeyboardPlannerExtraArgs "--planner planner/custom.onnx"
```
