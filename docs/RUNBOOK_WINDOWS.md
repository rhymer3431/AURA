# RUNBOOK (Windows Native Isaac Sim 4.2 + Agent + GR00T Policy)

## 1) Known paths

- Isaac Sim root: `C:\isaac-sim`
- Isaac Lab root: `C:\Users\mango\project\isaac-lab`
- Project root: `C:\Users\mango\project\isaac-aura`

`scripts/start_all.ps1` uses these paths by default.

## 2) Dependencies for GR00T policy client

Install in the Python environment used by `agent_runtime`:

```powershell
pip install numpy pyzmq msgpack
```

`manipulation.action_adapter.backend = "ros2_topic"`瑜??ъ슜??寃쎌슦 `rclpy`媛 ?꾩슂?⑸땲??
Windows?먯꽌??蹂댄넻 ROS2 ?ㅼ튂 ??ROS2 ?섍꼍?먯꽌 ?ㅽ뻾?댁빞 ?⑸땲??(`pip install rclpy`留뚯쑝濡쒕뒗 ?닿껐?섏? ?딅뒗 寃쎌슦媛 留롮쓬).

## 3) GR00T model download + FP8 TensorRT build

- Model: `nvidia/GR00T-N1.6-G1-PnPAppleToPlate`
- Embodiment tag: `UNITREE_G1`
- Local model dir: `models/gr00t_n1_6_g1_pnp_apple_to_plate`
- FP8 engine output: `models/gr00t_n1_6_g1_pnp_apple_to_plate/trt_fp8/dit_model_fp8.trt`

Run once from this repo root:

```powershell
pip install tensorrt diffusers peft tyro av dm-tree omegaconf msgpack-numpy
python scripts/prepare_groot_fp8.py --precision fp8
```

The command downloads the checkpoint, exports DiT ONNX, and builds the FP8 TensorRT engine.

The manipulator backend is now `gr00t_policy_server` by default. If policy server is unavailable, runtime falls back to mock mode (`fallback_to_mock=true`).

## 4) Start GR00T FP8 policy server

From this repository root:

```powershell
python scripts/run_groot_policy_server_fp8.py `
  --groot-repo-root C:\Users\mango\project\Isaac-GR00T-tmp `
  --model-path models/gr00t_n1_6_g1_pnp_apple_to_plate `
  --trt-engine-path models/gr00t_n1_6_g1_pnp_apple_to_plate/trt_fp8/dit_model_fp8.trt `
  --embodiment-tag UNITREE_G1 `
  --use-sim-policy-wrapper
```

> Note: `run_groot_policy_server_fp8.py` still depends on the Eagle backbone stack from Isaac-GR00T, which requires a FlashAttention-capable runtime. Use the Isaac-GR00T recommended Python/driver stack (typically Python 3.10 + CUDA-compatible flash-attn).

Then run this project:

```powershell
cd C:\Users\mango\project\isaac-aura
.\scripts\start_all.ps1 -NoInteractive
```

Optional: start GR00T server from `start_all.ps1` directly:

```powershell
.\scripts\start_all.ps1 `
  -StartGrootServer `
  -GrootServerCommand "python scripts/run_groot_policy_server_fp8.py --groot-repo-root C:\Users\mango\project\Isaac-GR00T-tmp --model-path models/gr00t_n1_6_g1_pnp_apple_to_plate --trt-engine-path models/gr00t_n1_6_g1_pnp_apple_to_plate/trt_fp8/dit_model_fp8.trt --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper"
```

## 5) Real Isaac Sim process (non-mock)

```powershell
.\scripts\start_all.ps1 -NoInteractive
```

`start_all.ps1` auto-detects Isaac Python at `C:\isaac-sim\python.bat`.
(`MockIsaac` default is now `false`.)

To force mock mode:

```powershell
.\scripts\start_all.ps1 -MockIsaac -NoInteractive
```

GUI濡?Isaac Sim 李쎌쓣 ?꾩썙 ?뺤씤?섎젮硫?

```powershell
.\scripts\start_all.ps1 -IsaacGui -NoInteractive
```

## 6) G1 action output routing

`apps/agent_runtime/config.yaml`:

- `manipulation.action_adapter.backend = "log"`: log predicted actions only
- `manipulation.action_adapter.backend = "ros2_topic"`: publish action vectors to ROS2 topics:
  - `/g1/cmd/joint_commands` (沅뚯옣, Isaac Sim ROS2 bridge articulation ?쒖뼱??
  - `/g1/cmd/left_arm`
  - `/g1/cmd/right_arm`
  - `/g1/cmd/left_hand`
  - `/g1/cmd/right_hand`
  - `/g1/cmd/waist`
  - `/g1/cmd/base_height`
  - `/g1/cmd/navigate`

## 7) Notes

- Current implementation executes policy outputs through an action adapter. Final low-level retargeting to Isaac articulation/joint controllers is environment-specific and remains a TODO.
- Sensor stream to manipulation is still minimal. For best policy quality, connect live RGB + robot state vectors to `GrootManipulator.update_state_vectors(...)` and image input path in task execution.

## 8) Start only G1 + GEAR-SONIC + GR00T (parallel)

This starts three processes together:
- Isaac Sim G1 runner (`apps/isaacsim_runner/run_headless.py`)
- GEAR-SONIC server (`sonic_policy_server.py`)
- GR00T policy server (`scripts/run_groot_policy_server_fp8.py` or config `start_command`)

```powershell
cd C:\Users\mango\project\isaac-aura
.\scripts\start_g1_sonic_groot.ps1
```

Optional examples:

```powershell
# Isaac Sim GUI
.\scripts\start_g1_sonic_groot.ps1 -IsaacGui

# Override GR00T startup command
.\scripts\start_g1_sonic_groot.ps1 `
  -GrootServerCommand "python scripts/run_groot_policy_server_fp8.py --groot-repo-root C:/Users/mango/project/Isaac-GR00T-tmp --model-path models/gr00t_n1_6_g1_pnp_apple_to_plate --trt-engine-path models/gr00t_n1_6_g1_pnp_apple_to_plate/trt_fp8/dit_model_fp8.trt --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper"
```

Press `Ctrl+C` to stop all three child processes.

## 9) Realtime GR00T command CLI

Directly send text commands to GR00T without planner/task decomposition:

```powershell
cd C:\Users\mango\project\isaac-aura
.\scripts\start_groot_realtime.ps1
```

CLI examples:

- `pick apple`
- `inspect apple`
- `move right arm to the plate`
- `exit`

One-shot mode:

```powershell
.\scripts\start_groot_realtime.ps1 -Command "pick apple" -NoInteractive
```
