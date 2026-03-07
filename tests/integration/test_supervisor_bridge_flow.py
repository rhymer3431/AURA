from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.g1_bridge import NavDPCommandSource
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate


class _FakeSimulationApp:
    def update(self) -> None:
        return None


class _FakeController:
    def get_base_state(self):
        return SimpleNamespace(
            position_w=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )


class _FakePlanningSession:
    def __init__(self) -> None:
        self.last_action_type = ""
        self._plan_version = -1

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

    def shutdown(self) -> None:
        return None


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
        detector_engine_path="",
    )


def test_supervisor_to_g1_bridge_command_flow() -> None:
    planning_session = _FakePlanningSession()
    command_source = NavDPCommandSource(_args(), planning_session=planning_session)
    command_source.initialize(_FakeSimulationApp(), stage=None, controller=_FakeController())
    command_source.update(1)

    assert planning_session.last_action_type == "NAV_TO_PLACE"
    assert command_source.supervisor.snapshot()["state"] == "GoToRememberedObject"
    assert command_source._active_command is not None
