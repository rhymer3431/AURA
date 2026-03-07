from __future__ import annotations

import inspect
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import runtime.planning_session as planning_module
from inference.navdp.base import NavDPNoGoalResponse, NavDPPointGoalResponse
from ipc.messages import ActionCommand
from runtime.planning_session import PlanningSession


class _FakeNavDPClient:
    def __init__(self) -> None:
        self.reset_calls = 0

    @property
    def backend_name(self) -> str:
        return "fake_navdp"

    def navigator_reset(self, intrinsic: np.ndarray, batch_size: int = 1) -> str:
        assert intrinsic.shape == (3, 3)
        assert batch_size == 1
        self.reset_calls += 1
        return self.backend_name

    def pointgoal_step(self, point_goals: np.ndarray, rgb_images: np.ndarray, depth_images_m: np.ndarray, sensor_meta=None):
        _ = sensor_meta
        assert point_goals.shape == (1, 2)
        assert rgb_images.shape[0] == 1
        assert depth_images_m.shape[0] == 1
        trajectory = np.asarray([[0.4, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)
        return NavDPPointGoalResponse(
            trajectory=trajectory,
            all_trajectory=trajectory.reshape(1, trajectory.shape[0], trajectory.shape[1]),
            all_values=np.ones((1, trajectory.shape[0]), dtype=np.float32),
            server_input_meta={"backend": self.backend_name},
        )

    def nogoal_step(self, rgb_images: np.ndarray, depth_images_m: np.ndarray) -> NavDPNoGoalResponse:
        assert rgb_images.shape[0] == 1
        assert depth_images_m.shape[0] == 1
        trajectory = np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32)
        return NavDPNoGoalResponse(
            trajectory=trajectory,
            all_trajectory=trajectory.reshape(1, trajectory.shape[0], trajectory.shape[1]),
            all_values=np.ones((1, trajectory.shape[0]), dtype=np.float32),
        )


def _args() -> Namespace:
    return Namespace(
        use_trajectory_z=False,
        plan_wait_timeout_sec=0.5,
        navdp_backend="heuristic",
        navdp_checkpoint="",
        navdp_device="cpu",
        navdp_amp=False,
        navdp_amp_dtype="float16",
        navdp_tf32=False,
        stop_threshold=-3.0,
    )


def test_planning_session_uses_local_executor_without_legacy_http() -> None:
    fake_client = _FakeNavDPClient()
    session = PlanningSession(_args(), navdp_client_factory=lambda intrinsic, args: fake_client)
    session.initialize_local(intrinsic=np.eye(3, dtype=np.float32))

    observation = session.build_local_observation(
        frame_id=1,
        rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        depth=np.ones((16, 16), dtype=np.float32),
        camera_pose_xyz=(0.0, 0.0, 1.2),
        camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    update = session.plan_with_observation(
        observation,
        action_command=ActionCommand(action_type="NAV_TO_POSE", target_pose_xyz=(1.0, 0.0, 0.0)),
        robot_pos_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    session.shutdown()

    assert fake_client.reset_calls == 1
    assert update.trajectory_world.shape == (2, 3)
    assert update.stats.successful_calls >= 1
    assert "legacy_http" not in inspect.getsource(planning_module)
