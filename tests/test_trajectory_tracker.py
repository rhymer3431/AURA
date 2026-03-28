from __future__ import annotations

import numpy as np

from control.trajectory_tracker import TrajectoryTracker, TrajectoryTrackerConfig


def _tracker() -> TrajectoryTracker:
    return TrajectoryTracker(TrajectoryTrackerConfig())


def _quat_from_yaw(yaw_rad: float) -> np.ndarray:
    return np.asarray([np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)], dtype=np.float32)


def test_tracker_commands_forward_velocity_for_straight_path():
    tracker = _tracker()
    tracker.set_trajectory(np.asarray([[0.5, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float32), plan_version=1, timestamp=0.0)

    result = tracker.compute_command(np.asarray([0.0, 0.0, 0.8], dtype=np.float32), _quat_from_yaw(0.0), now=0.1)

    assert result.command[0] > 0.1
    assert abs(float(result.command[1])) < 1.0e-4
    assert abs(float(result.command[2])) < 1.0e-4


def test_tracker_commands_positive_vy_for_left_offset_path():
    tracker = _tracker()
    tracker.set_trajectory(np.asarray([[0.0, 0.6, 0.0], [0.0, 1.2, 0.0]], dtype=np.float32), plan_version=1, timestamp=0.0)

    result = tracker.compute_command(np.asarray([0.0, 0.0, 0.8], dtype=np.float32), _quat_from_yaw(0.0), now=0.1)

    assert result.command[1] > 0.05


def test_tracker_slows_translation_for_large_heading_error():
    tracker = _tracker()
    tracker.set_trajectory(np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32), plan_version=1, timestamp=0.0)

    result = tracker.compute_command(np.asarray([0.0, 0.0, 0.8], dtype=np.float32), _quat_from_yaw(0.0), now=0.1)

    assert abs(float(result.command[0])) < 1.0e-4
    assert abs(float(result.command[1])) < 1.0e-4
    assert abs(float(result.command[2])) > 0.1


def test_tracker_returns_zero_for_stale_trajectory():
    tracker = _tracker()
    tracker.set_trajectory(np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32), plan_version=1, timestamp=0.0)

    result = tracker.compute_command(np.asarray([0.0, 0.0, 0.8], dtype=np.float32), _quat_from_yaw(0.0), now=2.0)

    assert result.stale is True
    assert np.allclose(result.command, np.zeros(3, dtype=np.float32))


def test_tracker_builds_pose_command_from_lookahead_target_and_segment_heading() -> None:
    tracker = _tracker()
    tracker.set_trajectory(
        np.asarray([[0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [1.5, 1.0, 0.0]], dtype=np.float32),
        plan_version=1,
        timestamp=0.0,
    )

    target = tracker.compute_target_pose(
        np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
        _quat_from_yaw(0.0),
        now=0.1,
    )

    assert target.stale is False
    assert target.target_idx >= 1
    np.testing.assert_allclose(target.pose_command_b[:2], np.asarray([1.0, 0.5], dtype=np.float32), atol=1.0e-4)
    assert abs(float(target.pose_command_b[2])) < 1.0e-6
    assert target.pose_command_b[3] > 0.0


def test_tracker_handoff_preserves_projected_progress_for_small_replan() -> None:
    tracker = _tracker()
    loop_traj = np.asarray(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.8, 0.8, 0.0], [0.0, 0.8, 0.0], [0.0, 0.05, 0.0]],
        dtype=np.float32,
    )
    tracker.set_trajectory(loop_traj, plan_version=1, timestamp=0.0, reset_progress=False, seed_progress_idx=4)

    handoff = tracker.compute_handoff(
        loop_traj + np.asarray([0.0, 0.02, 0.0], dtype=np.float32),
        position_w=np.asarray([0.0, 0.04, 0.8], dtype=np.float32),
    )

    assert handoff.reset_progress is False
    assert handoff.seed_progress_idx == 4
    tracker.set_trajectory(
        loop_traj + np.asarray([0.0, 0.02, 0.0], dtype=np.float32),
        plan_version=2,
        timestamp=0.1,
        reset_progress=handoff.reset_progress,
        seed_progress_idx=handoff.seed_progress_idx,
    )

    assert tracker.progress_idx == 4


def test_tracker_handoff_resets_progress_for_large_target_shift() -> None:
    tracker = TrajectoryTracker(TrajectoryTrackerConfig(lookahead_distance_m=0.1))
    tracker.set_trajectory(
        np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32),
        plan_version=1,
        timestamp=0.0,
    )

    handoff = tracker.compute_handoff(
        np.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=np.float32),
        position_w=np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
    )

    assert handoff.reset_progress is True
    assert handoff.seed_progress_idx is None
    assert handoff.target_shift_m > tracker.config.handoff_reset_distance_m


def test_tracker_handoff_resets_progress_for_large_heading_delta() -> None:
    tracker = TrajectoryTracker(TrajectoryTrackerConfig(lookahead_distance_m=0.1))
    tracker.set_trajectory(
        np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32),
        plan_version=1,
        timestamp=0.0,
    )

    handoff = tracker.compute_handoff(
        np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.4, 0.0]], dtype=np.float32),
        position_w=np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
    )

    assert handoff.reset_progress is True
    assert handoff.seed_progress_idx is None
    assert handoff.heading_delta_rad > tracker.config.handoff_reset_heading_rad
