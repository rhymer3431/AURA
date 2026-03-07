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
