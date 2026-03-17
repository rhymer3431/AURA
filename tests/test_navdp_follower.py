from __future__ import annotations

from pathlib import Path

import numpy as np

from control.navdp_follower import NavDPFollower, NavDPFollowerConfig
from locomotion.paths import resolve_default_follower_policy_path


ROOT = Path(__file__).resolve().parents[1]
FOLLOWER_POLICY_PATH = ROOT / "artifacts" / "models" / "navdp_follower.onnx"


class _FakePolicySession:
    def __init__(self, outputs: np.ndarray) -> None:
        self.input_shape = (1, 13)
        self.output_shape = (1, 3)
        self._outputs = np.asarray(outputs, dtype=np.float32).reshape(1, 3)

    def run(self, observation: np.ndarray) -> np.ndarray:
        self.last_observation = np.asarray(observation, dtype=np.float32).copy()
        return self._outputs.copy()

    def close(self) -> None:
        return None


def test_resolve_default_follower_policy_path_prefers_artifacts_model(tmp_path: Path) -> None:
    models_dir = tmp_path / "artifacts" / "models"
    legacy_dir = tmp_path / "src" / "control" / "models"
    models_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    preferred_path = models_dir / "navdp_follower.onnx"
    legacy_path = legacy_dir / "navdp_follower.onnx"
    preferred_path.write_bytes(b"preferred")
    legacy_path.write_bytes(b"legacy")

    assert resolve_default_follower_policy_path(str(tmp_path)) == str(preferred_path.resolve())


def test_resolve_default_follower_policy_path_falls_back_to_legacy_model(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "src" / "control" / "models"
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / "navdp_follower.onnx"
    legacy_path.write_bytes(b"legacy")

    assert resolve_default_follower_policy_path(str(tmp_path)) == str(legacy_path.resolve())


def test_follower_export_has_expected_io_shapes() -> None:
    follower = NavDPFollower(
        NavDPFollowerConfig(
            policy_path=str(FOLLOWER_POLICY_PATH),
            onnx_device="cpu",
        )
    )
    try:
        assert tuple(follower._policy_session.input_shape) == (1, 13)  # noqa: SLF001
        assert tuple(follower._policy_session.output_shape) == (1, 3)  # noqa: SLF001
    finally:
        follower.close()


def test_follower_builds_expected_observation_and_clips_output() -> None:
    session = _FakePolicySession(outputs=np.asarray([2.0, -1.0, 3.0], dtype=np.float32))
    follower = NavDPFollower(
        NavDPFollowerConfig(
            policy_path=str(FOLLOWER_POLICY_PATH),
            onnx_device="cpu",
            max_vx=0.5,
            max_vy=0.3,
            max_wz=0.8,
        ),
        policy_session=session,
    )
    result = follower.compute_command(
        pose_command_b=np.asarray([1.0, -2.0, 0.0, 0.3], dtype=np.float32),
        base_lin_vel_w=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        base_ang_vel_w=np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert result.observation.shape == (13,)
    np.testing.assert_allclose(result.observation[:3], np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(result.observation[3:6], np.asarray([0.4, 0.5, 0.6], dtype=np.float32))
    np.testing.assert_allclose(result.observation[6:9], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(result.observation[9:], np.asarray([1.0, -2.0, 0.0, 0.3], dtype=np.float32))
    np.testing.assert_allclose(result.command, np.asarray([0.5, -0.3, 0.8], dtype=np.float32))
