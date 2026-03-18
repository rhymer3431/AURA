from __future__ import annotations

from argparse import Namespace

import numpy as np

from ipc.messages import ActionCommand
from locomotion.worker import LocomotionWorker
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate


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
        obstacle_hold_distance_m=0.70,
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
    return ActionCommand(action_type="LOCAL_SEARCH", task_id="planner_managed", metadata={"planner_managed": True})


def _observation(depth: np.ndarray | None = None) -> ExecutionObservation:
    depth_image = np.ones((18, 18), dtype=np.float32) if depth is None else np.asarray(depth, dtype=np.float32)
    return ExecutionObservation(
        frame_id=1,
        rgb=np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8),
        depth=depth_image,
        sensor_meta={},
        cam_pos=np.asarray([0.0, 0.0, 1.2], dtype=np.float32),
        cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
    )


def test_locomotion_worker_returns_proposal_without_final_status() -> None:
    worker = LocomotionWorker(_args())
    proposal = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=TrajectoryUpdate(
            trajectory_world=np.asarray([[0.8, 0.0, 0.0], [1.2, 0.4, 0.0]], dtype=np.float32),
            plan_version=5,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
            source_frame_id=1,
            stop=False,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(
        proposal.command_vector,
        np.asarray([0.5, 0.3, 1.5 * np.arctan2(0.4, 1.2)], dtype=np.float32),
        atol=1.0e-4,
    )
    assert proposal.evaluation.reached_goal is False
    assert not hasattr(proposal, "status")


def test_locomotion_worker_reports_obstacle_metadata_only() -> None:
    worker = LocomotionWorker(_args())
    depth = np.full((18, 18), 1.2, dtype=np.float32)
    depth[:, 7:11] = 0.25
    depth[:, 11:15] = 2.0
    proposal = worker.execute(
        frame_idx=1,
        observation=_observation(depth),
        action_command=_planner_command(),
        trajectory_update=TrajectoryUpdate(
            trajectory_world=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            plan_version=6,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
            source_frame_id=1,
            stop=False,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert float(proposal.command_vector[0]) < 0.0
    assert proposal.metadata["obstacle_defense"] is True
    assert proposal.metadata["obstacle_defense_mode"] == "backoff_turn"
