from __future__ import annotations

import socket
import sys
import time
import uuid
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter
from apps.runtime_common import build_runtime_io
from runtime.aura_runtime import AuraRuntimeCommandSource
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate


class _FakeSimulationApp:
    def update(self) -> None:
        return None


class _FakeController:
    def get_base_state(self):
        return SimpleNamespace(
            position_w=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            lin_vel_w=np.zeros(3, dtype=np.float32),
            ang_vel_w=np.zeros(3, dtype=np.float32),
        )


class _FakePlanningSession:
    def __init__(self) -> None:
        self.last_action_type = ""
        self._plan_version = -1
        self.started_instructions: list[str] = []
        self.health_checks: list[str] = []
        self._viewer_trajectory_world = [
            [0.0, 0.0, 2.0],
            [0.4, 0.0, 2.0],
            [0.8, 0.0, 2.0],
        ]

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        _ = simulation_app, stage

    def capture_observation(self, frame_id: int, env=None):  # noqa: ANN001
        _ = env
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        rgb[24:72, 28:68, 0] = 255
        depth = np.full((96, 96), 1.5, dtype=np.float32)
        intrinsic = np.asarray([[96.0, 0.0, 48.0], [0.0, 96.0, 48.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return ExecutionObservation(
            frame_id=int(frame_id),
            rgb=rgb,
            depth=depth,
            sensor_meta={"target_class_hint": "apple", "color_hint": "red", "room_id": "kitchen"},
            cam_pos=np.asarray([0.0, 0.0, 1.2], dtype=np.float32),
            cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            intrinsic=intrinsic,
        )

    def plan_with_observation(self, observation, *, action_command, robot_pos_world, robot_yaw, robot_quat_wxyz):  # noqa: ANN001
        _ = observation, robot_pos_world, robot_yaw, robot_quat_wxyz
        self.last_action_type = action_command.action_type if action_command is not None else ""
        self._plan_version += 1
        trajectory = np.asarray([[0.3, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float32)
        return TrajectoryUpdate(
            trajectory_world=trajectory,
            plan_version=self._plan_version,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=2.0, last_plan_step=int(observation.frame_id)),
            source_frame_id=int(observation.frame_id),
            action_command=action_command,
            stop=False,
        )

    def ensure_navdp_service_ready(self, *, context: str) -> None:
        self.health_checks.append(f"navdp:{context}")

    def ensure_dual_service_ready(self, *, context: str) -> None:
        self.health_checks.append(f"dual:{context}")

    def start_dual_task(self, instruction: str) -> None:
        self.started_instructions.append(str(instruction))

    def viewer_overlay_state(self) -> dict[str, object]:
        return {
            "trajectory_world": list(self._viewer_trajectory_world),
            "plan_version": 4,
            "goal_version": 2,
            "traj_version": 3,
        }

    def shutdown(self) -> None:
        return None

    def active_memory_instruction(self) -> str:
        return self.started_instructions[-1] if self.started_instructions else ""


class _InteractivePlanningSession(_FakePlanningSession):
    def __init__(self) -> None:
        super().__init__()
        self.submitted_instructions: list[str] = []

    def submit_interactive_instruction(self, instruction: str) -> int:
        self.submitted_instructions.append(str(instruction))
        return len(self.submitted_instructions)

    def cancel_interactive_task(self) -> bool:
        return True

    def active_memory_instruction(self) -> str:
        return self.submitted_instructions[-1] if self.submitted_instructions else ""

    def plan_with_observation(self, observation, *, action_command, robot_pos_world, robot_yaw, robot_quat_wxyz):  # noqa: ANN001
        _ = robot_pos_world, robot_yaw, robot_quat_wxyz
        self.last_action_type = action_command.action_type if action_command is not None else ""
        self._plan_version += 1
        return TrajectoryUpdate(
            trajectory_world=np.asarray([[0.25, 0.0, 0.0], [0.55, 0.0, 0.0]], dtype=np.float32),
            plan_version=self._plan_version,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=2.0, last_plan_step=int(observation.frame_id)),
            source_frame_id=int(observation.frame_id),
            action_command=action_command,
            stop=False,
            interactive_phase="task_active",
            interactive_command_id=max(len(self.submitted_instructions), 1),
            interactive_instruction=self.submitted_instructions[-1] if self.submitted_instructions else "",
        )


def _args() -> Namespace:
    return Namespace(
        planner_mode="dual",
        instruction="아까 봤던 사과를 찾아가",
        spawn_demo_object=False,
        goal_x=None,
        goal_y=None,
        goal_tolerance_m=0.4,
        startup_updates=0,
        cmd_max_vx=0.5,
        cmd_max_vy=0.3,
        cmd_max_wz=0.8,
        lookahead_distance_m=0.6,
        heading_slowdown_rad=0.6,
        traj_stale_timeout_sec=1.5,
        cmd_accel_limit=1.0,
        cmd_yaw_accel_limit=1.5,
        log_interval=1,
        interactive_idle_log_interval=120,
        memory_db_path="state/memory/memory.sqlite",
        detector_model_path="artifacts/models/__missing__.engine",
        detector_device="",
    )


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_dual_mode_bootstraps_planner_managed_command_flow() -> None:
    planning_session = _FakePlanningSession()
    command_source = AuraRuntimeCommandSource(_args(), planning_session=planning_session)
    command_source.initialize(_FakeSimulationApp(), stage=None, controller=_FakeController())
    command_source.update(1)

    assert planning_session.started_instructions == ["아까 봤던 사과를 찾아가"]
    assert planning_session.health_checks == ["navdp:dual startup", "dual:dual startup"]
    assert planning_session.last_action_type == "LOCAL_SEARCH"
    assert command_source._active_command is not None
    assert command_source._active_command.metadata["planner_managed"] is True
    assert command_source._runtime_io is not None
    assert command_source._runtime_io.shm_ring is None


def test_interactive_mode_uses_planning_session_control_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    planning_session = _InteractivePlanningSession()
    args = _args()
    args.planner_mode = "interactive"
    args.instruction = ""
    monkeypatch.setattr(AuraRuntimeCommandSource, "_interactive_input_loop", lambda self: None)
    command_source = AuraRuntimeCommandSource(args, planning_session=planning_session)

    command_source.initialize(_FakeSimulationApp(), stage=None, controller=_FakeController())
    command_source.planning_session.submit_interactive_instruction("go to the loading dock")
    command_source.update(1)

    assert planning_session.health_checks == ["navdp:interactive startup"]
    assert planning_session.submitted_instructions == ["go to the loading dock"]
    assert planning_session.last_action_type == "LOCAL_SEARCH"
    assert command_source.supervisor.snapshot()["state"] == "Idle"
    assert tuple(np.round(command_source.command(), 4)) != (0.0, 0.0, 0.0)


def test_g1_view_mode_publishes_overlay_metadata_over_zmq_and_shm() -> None:
    pytest.importorskip("zmq")
    planning_session = _FakePlanningSession()
    port = _free_tcp_port()
    args = _args()
    args.launch_mode = "g1_view"
    args.resolved_launch_mode = "g1_view"
    args.headless = True
    args.viewer_control_endpoint = f"tcp://127.0.0.1:{port}"
    args.viewer_telemetry_endpoint = f"tcp://127.0.0.1:{port + 1}"
    args.viewer_shm_name = f"g1_view_test_{uuid.uuid4().hex[:12]}"
    args.viewer_shm_slot_size = 2 * 1024 * 1024
    args.viewer_shm_capacity = 4
    command_source = AuraRuntimeCommandSource(args, planning_session=planning_session)
    viewer_io = None
    try:
        command_source.initialize(_FakeSimulationApp(), stage=None, controller=_FakeController())
        viewer_io = build_runtime_io(
            bus_kind="zmq",
            endpoint=args.viewer_control_endpoint,
            bind=False,
            shm_name=args.viewer_shm_name,
            shm_slot_size=args.viewer_shm_slot_size,
            shm_capacity=args.viewer_shm_capacity,
            create_shm=False,
            role="agent",
            control_endpoint=args.viewer_control_endpoint,
            telemetry_endpoint=args.viewer_telemetry_endpoint,
            identity="test-g1-viewer",
        )
        time.sleep(0.1)

        observed_header = None
        for frame_idx in range(1, 8):
            command_source.update(frame_idx)
            deadline = time.time() + 0.25
            while time.time() < deadline:
                records = viewer_io.bus.poll("isaac.observation", max_items=32)
                if records:
                    observed_header = records[-1].message
                    break
                time.sleep(0.01)
            if observed_header is not None:
                break

        assert observed_header is not None
        assert "viewer_overlay" in observed_header.metadata
        overlay = observed_header.metadata["viewer_overlay"]
        assert overlay["detector_backend"] != ""
        assert overlay["detections"]
        assert overlay["detections"][0]["class_name"] == "apple"
        assert overlay["detections"][0]["bbox_xyxy"] == [28, 24, 67, 71]
        assert overlay["trajectory_pixels"] == [[48, 48], [96, 48], [144, 48]]
        assert overlay["plan_version"] == 4
        assert overlay["goal_version"] == 2
        assert overlay["traj_version"] == 3

        batch = IsaacBridgeAdapter(viewer_io.bus, shm_ring=viewer_io.shm_ring).reconstruct_batch(observed_header)
        assert batch.rgb_image is not None
        assert batch.depth_image_m is not None
        assert batch.rgb_image.shape == (96, 96, 3)
        assert planning_session.last_action_type == "LOCAL_SEARCH"
    finally:
        if viewer_io is not None:
            viewer_io.close()
        command_source.shutdown()
