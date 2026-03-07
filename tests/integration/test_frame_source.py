from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.sensors.isaac_live_source import IsaacLiveFrameSource
from apps.runtime_common import build_frame_source


class _FakeSensor:
    def __init__(self) -> None:
        self.intrinsic = np.asarray([[96.0, 0.0, 48.0], [0.0, 96.0, 48.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.last_capture_meta = {"rgb_source": "fake", "depth_source": "fake"}

    def capture_rgbd_with_meta(self, env):  # noqa: ANN001
        _ = env
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        depth = np.full((96, 96), 1.25, dtype=np.float32)
        return rgb, depth, {"rgb_source": "fake", "depth_source": "fake", "fallback_used": False}

    def get_rgb_camera_pose_world(self):
        return np.asarray([0.0, 0.0, 1.2], dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def test_auto_frame_source_uses_synthetic_fallback_when_live_unavailable() -> None:
    source = build_frame_source(mode="auto", scene="apple", source_name="test")

    report = source.start()
    sample = source.read()

    assert report.fallback_used is True
    assert sample is not None
    assert sample.source_name.endswith("_synthetic")
    source.close()


def test_live_frame_source_reuses_initialized_sensor_adapter() -> None:
    source = IsaacLiveFrameSource(sensor_adapter=_FakeSensor(), robot_pose_provider=lambda: (1.0, 2.0, 0.0), robot_yaw_provider=lambda: 0.3)

    report = source.start()
    sample = source.read()

    assert report.status == "ready"
    assert sample is not None
    assert sample.robot_pose_xyz == (1.0, 2.0, 0.0)
    assert sample.robot_yaw_rad == 0.3
    assert sample.camera_intrinsic.shape == (3, 3)
