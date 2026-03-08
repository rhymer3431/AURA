from __future__ import annotations

from argparse import Namespace

import numpy as np

from ipc.messages import ActionCommand
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate
from runtime.subgoal_executor import SubgoalExecutor


def _args() -> Namespace:
    return Namespace(
        cmd_max_vx=0.5,
        cmd_max_vy=0.3,
        cmd_max_wz=0.8,
        lookahead_distance_m=0.6,
        heading_slowdown_rad=0.6,
        traj_stale_timeout_sec=1.5,
        cmd_accel_limit=1.0,
        cmd_yaw_accel_limit=1.5,
    )


def _planner_command() -> ActionCommand:
    return ActionCommand(
        action_type="LOCAL_SEARCH",
        task_id="planner_managed",
        metadata={"planner_managed": True},
    )


def _observation() -> ExecutionObservation:
    return ExecutionObservation(
        frame_id=1,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        sensor_meta={},
        cam_pos=np.asarray([0.0, 0.0, 1.2], dtype=np.float32),
        cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
    )


class _FakePlanningSession:
    def __init__(self, update: TrajectoryUpdate) -> None:
        self._update = update

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        _ = simulation_app, stage

    def shutdown(self) -> None:
        return None

    def plan_with_observation(self, observation, *, action_command, robot_pos_world, robot_yaw, robot_quat_wxyz):  # noqa: ANN001
        _ = observation, action_command, robot_pos_world, robot_yaw, robot_quat_wxyz
        return self._update


def test_planner_managed_stop_maps_to_success() -> None:
    update = TrajectoryUpdate(
        trajectory_world=np.zeros((0, 3), dtype=np.float32),
        plan_version=1,
        stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
        source_frame_id=1,
        stop=True,
    )
    executor = SubgoalExecutor(_args(), planning_session=_FakePlanningSession(update))

    result = executor.step(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert result.evaluation.reached_goal is True
    assert result.status is not None
    assert result.status.state == "succeeded"


def test_planner_managed_error_maps_to_failure() -> None:
    update = TrajectoryUpdate(
        trajectory_world=np.zeros((0, 3), dtype=np.float32),
        plan_version=2,
        stats=PlannerStats(failed_calls=1, last_error="dual_step failed", last_plan_step=1),
        source_frame_id=1,
        stop=False,
    )
    executor = SubgoalExecutor(_args(), planning_session=_FakePlanningSession(update))

    result = executor.step(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert result.evaluation.reached_goal is False
    assert result.status is not None
    assert result.status.state == "failed"
    assert result.status.reason == "dual_step failed"
