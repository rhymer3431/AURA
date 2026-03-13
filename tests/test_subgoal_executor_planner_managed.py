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
        obstacle_defense_enabled=True,
        obstacle_stop_distance_m=0.45,
        obstacle_turn_distance_m=0.70,
        obstacle_side_bias_m=0.10,
        obstacle_min_valid_fraction=0.05,
        obstacle_min_turn_wz=0.35,
        obstacle_forward_trigger_mps=0.05,
        obstacle_slow_forward_vx_mps=0.08,
        obstacle_backoff_vx_mps=0.18,
        obstacle_lateral_nudge_vy_mps=0.12,
        obstacle_recovery_hold_sec=0.75,
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


def _observation_with_depth(depth: np.ndarray) -> ExecutionObservation:
    return ExecutionObservation(
        frame_id=1,
        rgb=np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8),
        depth=np.asarray(depth, dtype=np.float32),
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


def test_planner_managed_wait_holds_position() -> None:
    update = TrajectoryUpdate(
        trajectory_world=np.zeros((0, 3), dtype=np.float32),
        plan_version=3,
        stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
        source_frame_id=1,
        stop=False,
        planner_control_mode="wait",
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

    assert np.allclose(result.command_vector, np.zeros(3, dtype=np.float32))
    assert result.status is not None
    assert result.status.state == "running"


def test_planner_managed_yaw_delta_reaches_target() -> None:
    update = TrajectoryUpdate(
        trajectory_world=np.zeros((0, 3), dtype=np.float32),
        plan_version=4,
        stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
        source_frame_id=1,
        stop=False,
        planner_control_mode="yaw_delta",
        planner_yaw_delta_rad=float(np.pi / 6.0),
    )
    executor = SubgoalExecutor(_args(), planning_session=_FakePlanningSession(update))

    first = executor.step(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    second = executor.step(
        frame_idx=2,
        observation=_observation(),
        action_command=_planner_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=float(np.pi / 6.0),
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert first.command_vector[2] > 0.0
    assert first.status is not None
    assert first.status.state == "running"
    assert second.status is not None
    assert second.status.state == "succeeded"


def test_obstacle_defense_turns_toward_clearer_side() -> None:
    update = TrajectoryUpdate(
        trajectory_world=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        plan_version=5,
        stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
        source_frame_id=1,
        stop=False,
    )
    executor = SubgoalExecutor(_args(), planning_session=_FakePlanningSession(update))
    depth = np.full((18, 18), 1.2, dtype=np.float32)
    depth[:, 7:11] = 0.25
    depth[:, 11:15] = 2.0

    result = executor.step(
        frame_idx=1,
        observation=_observation_with_depth(depth),
        action_command=_planner_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert float(result.command_vector[0]) < 0.0
    assert float(result.command_vector[1]) < 0.0
    assert float(result.command_vector[2]) < 0.0
    assert result.status is not None
    assert result.status.metadata["obstacle_defense"] is True
    assert result.status.metadata["obstacle_defense_mode"] == "backoff_turn"


def test_obstacle_defense_backoffs_when_both_sides_are_blocked() -> None:
    update = TrajectoryUpdate(
        trajectory_world=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        plan_version=6,
        stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
        source_frame_id=1,
        stop=False,
    )
    executor = SubgoalExecutor(_args(), planning_session=_FakePlanningSession(update))
    depth = np.full((18, 18), 0.20, dtype=np.float32)

    result = executor.step(
        frame_idx=1,
        observation=_observation_with_depth(depth),
        action_command=_planner_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert float(result.command_vector[0]) < 0.0
    assert abs(float(result.command_vector[1])) < 1.0e-4
    assert abs(float(result.command_vector[2])) < 1.0e-4
    assert result.status is not None
    assert result.status.metadata["obstacle_defense"] is True
    assert result.status.metadata["obstacle_defense_mode"] == "backoff_hold"


def test_obstacle_defense_holds_recovery_briefly_after_contact() -> None:
    update = TrajectoryUpdate(
        trajectory_world=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        plan_version=7,
        stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
        source_frame_id=1,
        stop=False,
    )
    executor = SubgoalExecutor(_args(), planning_session=_FakePlanningSession(update))
    blocked_depth = np.full((18, 18), 1.2, dtype=np.float32)
    blocked_depth[:, 7:11] = 0.20
    blocked_depth[:, 11:15] = 2.0
    clear_depth = np.full((18, 18), 2.5, dtype=np.float32)

    first = executor.step(
        frame_idx=1,
        observation=_observation_with_depth(blocked_depth),
        action_command=_planner_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    second = executor.step(
        frame_idx=2,
        observation=_observation_with_depth(clear_depth),
        action_command=_planner_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert float(first.command_vector[0]) < 0.0
    assert float(second.command_vector[0]) < 0.0
    assert second.status is not None
    assert second.status.metadata["obstacle_defense_mode"] == "backoff_recovery"
    assert second.status.metadata["obstacle_recovery_active"] is True


def test_obstacle_defense_slow_turn_adds_centering_bias() -> None:
    update = TrajectoryUpdate(
        trajectory_world=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        plan_version=8,
        stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
        source_frame_id=1,
        stop=False,
    )
    executor = SubgoalExecutor(_args(), planning_session=_FakePlanningSession(update))
    depth = np.full((18, 18), 1.5, dtype=np.float32)
    depth[:, 7:11] = 0.60
    depth[:, 11:15] = 1.8

    result = executor.step(
        frame_idx=1,
        observation=_observation_with_depth(depth),
        action_command=_planner_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert 0.0 <= float(result.command_vector[0]) <= 0.08 + 1.0e-4
    assert float(result.command_vector[1]) < 0.0
    assert float(result.command_vector[2]) < 0.0
    assert result.status is not None
    assert result.status.metadata["obstacle_defense_mode"] == "slow_turn"
