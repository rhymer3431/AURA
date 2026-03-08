from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.sensors.d455_sensor import D455SensorAdapter, D455SensorAdapterConfig


class _FakeCamera:
    def __init__(self, frame: dict[str, object]) -> None:
        self._frame = dict(frame)

    def get_distance_to_camera(self):
        raise RuntimeError("force current_frame path")

    def get_current_frame(self) -> dict[str, object]:
        return dict(self._frame)


class _FakePrim:
    def __init__(self, path: str) -> None:
        self._path = str(path)

    def IsValid(self) -> bool:
        return True

    def GetPath(self) -> str:
        return self._path


class _FakeStage:
    def __init__(self, paths: list[str]) -> None:
        self._paths = [str(path) for path in paths]

    def Traverse(self):
        return [_FakePrim(path) for path in self._paths]


def test_extract_distance_to_camera_annotator_uses_replicator_output() -> None:
    sensor = D455SensorAdapter(D455SensorAdapterConfig())
    camera = _FakeCamera(
        {
            "distance_to_camera": np.full((4, 4), 2.5, dtype=np.float32),
            "distance_to_image_plane": np.full((4, 4), 1.0, dtype=np.float32),
            "depth": np.full((4, 4), 0.5, dtype=np.float32),
        }
    )

    depth = sensor._extract_distance_to_camera_annotator(camera)  # noqa: SLF001

    assert depth is not None
    assert depth.shape == (4, 4)
    assert np.allclose(depth, 2.5)


def test_discover_camera_prims_ignores_pseudo_depth_camera() -> None:
    stage = _FakeStage(
        [
            "/World/G1/torso_link/d435_link/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
            "/World/G1/torso_link/d435_link/Realsense/RSD455/Camera_Pseudo_Depth",
        ]
    )

    rgb_path, depth_path = D455SensorAdapter._discover_camera_prims(stage)  # noqa: SLF001

    assert rgb_path == "/World/G1/torso_link/d435_link/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
    assert depth_path is None


def test_capture_rgbd_with_meta_does_not_fallback_to_env_depth_when_rgb_exists() -> None:
    sensor = D455SensorAdapter(D455SensorAdapterConfig())
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    sensor._capture_rgb = lambda: rgb  # noqa: SLF001
    sensor._capture_depth_from_camera = lambda: (None, "missing")  # noqa: SLF001
    sensor._capture_depth_from_env = lambda _env: np.full((8, 8), 1.0, dtype=np.float32)  # noqa: SLF001

    captured_rgb, captured_depth, meta = sensor.capture_rgbd_with_meta(object())

    assert captured_rgb is rgb
    assert captured_depth is None
    assert meta["depth_source"] == "missing"
    assert meta["note"] == "Depth capture unavailable."
