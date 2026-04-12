from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np

from systems.control.runtime.runtime_controller import InternVlaNavDpController


class _FakeRobot:
    def __init__(self):
        self.base_pos = np.asarray((0.0, 0.0, 0.8), dtype=np.float32)
        self.base_quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)

    def get_world_pose(self):
        return self.base_pos.copy(), self.base_quat.copy()

    def get_linear_velocity(self):
        return np.asarray((0.0, 0.0, 0.0), dtype=np.float32)

    def get_angular_velocity(self):
        return np.asarray((0.0, 0.0, 0.0), dtype=np.float32)


class _FakeController:
    def __init__(self):
        self.robot = _FakeRobot()


def _args():
    return SimpleNamespace(
        navigation_url="http://nav",
        navigation_update_hz=5.0,
        navigation_trajectory_timeout=0.2,
        lookahead_distance=0.5,
        vx_max=0.5,
        vy_max=0.3,
        wz_max=1.0,
        cmd_smoothing_tau=0.1,
    )


def test_control_follower_holds_zero_for_empty_stale_and_error_payloads() -> None:
    controller = InternVlaNavDpController(_args())
    controller.bind_controller(_FakeController())

    assert np.allclose(controller.command(), np.zeros(3, dtype=np.float32))
    assert controller.runtime_status()["state_label"] == "waiting"

    controller.update_navigation_payload(
        {
            "status": "error",
            "trajectory_world_xy": [],
            "stamp_s": time.monotonic(),
            "last_error": "boom",
        }
    )
    assert np.allclose(controller.command(), np.zeros(3, dtype=np.float32))
    assert controller.runtime_status()["state_label"] == "error"

    controller.update_navigation_payload(
        {
            "status": "running",
            "trajectory_world_xy": [[0.5, 0.0], [1.0, 0.0]],
            "stamp_s": time.monotonic() - 1.0,
            "last_error": None,
        }
    )
    assert np.allclose(controller.command(), np.zeros(3, dtype=np.float32))
    assert controller.runtime_status()["state_label"] == "stale"


def test_control_follower_tracks_fresh_trajectory() -> None:
    controller = InternVlaNavDpController(_args())
    controller.bind_controller(_FakeController())
    controller.update_navigation_payload(
        {
            "status": "running",
            "trajectory_world_xy": [[0.5, 0.0], [1.0, 0.0]],
            "stamp_s": time.monotonic(),
            "last_error": None,
        }
    )

    command = controller.command()

    assert command.shape == (3,)
    assert float(np.linalg.norm(command)) > 0.0
    assert controller.runtime_status()["state_label"] == "tracking"


def test_control_follower_executes_direct_action_overrides_from_system2() -> None:
    controller = InternVlaNavDpController(_args())
    fake_controller = _FakeController()
    controller.bind_controller(fake_controller)
    controller.update_navigation_payload(
        {
            "status": "running",
            "trajectory_world_xy": [],
            "stamp_s": time.monotonic(),
            "last_error": None,
            "system2": {
                "status": "hold",
                "decision_mode": "yaw_right",
                "action_sequence": ["yaw_right"],
            },
        }
    )

    command = controller.command()

    assert command.shape == (3,)
    assert command[2] < 0.0
    status = controller.runtime_status()
    assert status["action_override_mode"] == "yaw_right"
    assert status["system2_decision_mode"] == "yaw_right"
    assert status["state_label"] == "yaw-right-override"


def test_control_follower_clears_direct_action_when_trajectory_arrives() -> None:
    controller = InternVlaNavDpController(_args())
    fake_controller = _FakeController()
    controller.bind_controller(fake_controller)
    controller.update_navigation_payload(
        {
            "status": "running",
            "trajectory_world_xy": [],
            "stamp_s": time.monotonic(),
            "last_error": None,
            "system2": {
                "status": "hold",
                "decision_mode": "yaw_left",
                "action_sequence": ["yaw_left"],
            },
        }
    )
    first_command = controller.command()
    assert first_command[2] > 0.0

    controller.update_navigation_payload(
        {
            "status": "running",
            "trajectory_world_xy": [[0.5, 0.0], [1.0, 0.0]],
            "stamp_s": time.monotonic(),
            "last_error": None,
            "system2": {
                "status": "goal",
                "decision_mode": "pixel_goal",
                "action_sequence": [],
            },
        }
    )

    second_command = controller.command()

    assert second_command.shape == (3,)
    assert float(np.linalg.norm(second_command)) > 0.0
    assert controller.runtime_status()["action_override_mode"] is None
    assert controller.runtime_status()["state_label"] == "tracking"
