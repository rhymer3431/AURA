from __future__ import annotations

import sys
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from systems.transport.bus.inproc_bus import InprocBus
from systems.transport.messages import ActionCommand
from runtime.frame_bridge_runtime import FrameBridgeCommandSource
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


class _FakeSensor:
    def __init__(self) -> None:
        self.intrinsic = np.asarray([[96.0, 0.0, 48.0], [0.0, 96.0, 48.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.last_capture_meta = {
            "rgb_source": "fake_rgb",
            "depth_source": "fake_depth",
            "fallback_used": False,
            "runtime_mount": True,
            "camera_prim_path": "/World/G1/CameraRGB",
            "depth_camera_prim_path": "/World/G1/CameraDepth",
        }
        self.rgb_prim_path = "/World/G1/CameraRGB"
        self.depth_prim_path = "/World/G1/CameraDepth"
        self.runtime_camera_mode = True

    def capture_rgbd_with_meta(self, env):  # noqa: ANN001
        _ = env
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        rgb[16:72, 20:76, 0] = 255
        depth = np.full((96, 96), 1.25, dtype=np.float32)
        return rgb, depth, dict(self.last_capture_meta)

    def get_rgb_camera_pose_world(self):
        return np.asarray([0.0, 0.0, 1.2], dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _FakePlanningSession:
    def __init__(self) -> None:
        self.sensor = _FakeSensor()
        self.last_sensor_init_report = {
            "ok": True,
            "message": "fake sensor ready",
            "camera_prim_path": self.sensor.rgb_prim_path,
            "depth_camera_prim_path": self.sensor.depth_prim_path,
            "runtime_mount": True,
        }
        self.last_action_type = ""
        self._plan_version = -1

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        _ = simulation_app, stage

    def capture_observation(self, frame_id: int, env=None):  # noqa: ANN001
        _ = frame_id, env
        return None

    def build_local_observation(
        self,
        *,
        frame_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_pose_xyz,
        camera_quat_wxyz,
        intrinsic=None,
        sensor_meta=None,
    ) -> ExecutionObservation:
        return ExecutionObservation(
            frame_id=int(frame_id),
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            sensor_meta=dict(sensor_meta or {}),
            cam_pos=np.asarray(camera_pose_xyz, dtype=np.float32),
            cam_quat=np.asarray(camera_quat_wxyz, dtype=np.float32),
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
        )

    def plan_with_observation(self, observation, *, action_command, robot_pos_world, robot_yaw, robot_quat_wxyz):  # noqa: ANN001
        _ = robot_pos_world, robot_yaw, robot_quat_wxyz
        self.last_action_type = action_command.action_type if action_command is not None else ""
        self._plan_version += 1
        return TrajectoryUpdate(
            trajectory_world=np.asarray([[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
            plan_version=self._plan_version,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=int(observation.frame_id)),
            source_frame_id=int(observation.frame_id),
            action_command=action_command,
            stop=False,
        )

    def shutdown(self) -> None:
        return None


def _args(tmp_path: Path | None = None) -> Namespace:
    return Namespace(
        command="",
        startup_updates=0,
        strict_live=False,
        image_width=96,
        image_height=96,
        depth_max_m=5.0,
        cmd_max_vx=0.5,
        cmd_max_vy=0.3,
        cmd_max_wz=0.8,
        lookahead_distance_m=0.6,
        heading_slowdown_rad=0.6,
        traj_stale_timeout_sec=1.5,
        cmd_accel_limit=1.0,
        cmd_yaw_accel_limit=1.5,
        log_interval=1,
        physics_dt=1.0 / 60.0,
        sensor_report_path="" if tmp_path is None else str(tmp_path / "sensor_report.json"),
    )


def test_frame_bridge_command_source_publishes_live_frame_and_status(tmp_path: Path) -> None:
    bus = InprocBus()
    planning_session = _FakePlanningSession()
    command_source = FrameBridgeCommandSource(_args(tmp_path), bus=bus, planning_session=planning_session)

    command_source.initialize(_FakeSimulationApp(), stage=None, controller=_FakeController())
    bus.publish("isaac.command", ActionCommand(action_type="NAV_TO_POSE", target_pose_xyz=(1.0, 0.0, 0.0)))

    command_source.update(1)

    capability_records = bus.poll("isaac.capability", max_items=4)
    notice_records = bus.poll("isaac.notice", max_items=8)
    observation_records = bus.poll("isaac.observation", max_items=4)
    status_records = bus.poll("isaac.status", max_items=4)

    assert capability_records[0].message.component == "sensor"
    assert notice_records
    assert observation_records
    assert status_records == []

    frame_header = observation_records[-1].message
    assert frame_header.robot_pose_xyz == (0.0, 0.0, 0.0)
    assert tuple(round(float(v), 4) for v in frame_header.camera_pose_xyz) == (0.0, 0.0, 1.2)
    assert frame_header.sim_time_s >= 0.0
    assert frame_header.metadata["capture_report"]["camera_prim_path"] == "/World/G1/CameraRGB"
    assert tuple(np.round(command_source.command(), 4)) == (0.0, 0.0, 0.0)
    report_payload = json.loads((tmp_path / "sensor_report.json").read_text(encoding="utf-8"))
    assert report_payload["status"] == "ready"
    assert report_payload["details"]["camera_prim_path"] == "/World/G1/CameraRGB"
