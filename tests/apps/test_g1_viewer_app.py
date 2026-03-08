from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apps import g1_viewer_app


class _Cv2ProbeFailure:
    def __init__(self) -> None:
        self.imshow = lambda *_args, **_kwargs: None
        self.waitKey = lambda *_args, **_kwargs: 0
        self.rectangle = lambda *_args, **_kwargs: None
        self.putText = lambda *_args, **_kwargs: None
        self.polylines = lambda *_args, **_kwargs: None
        self.circle = lambda *_args, **_kwargs: None
        self.namedWindow = self._named_window
        self.destroyWindow = lambda *_args, **_kwargs: None

    @staticmethod
    def _named_window(*_args, **_kwargs):
        raise RuntimeError("The function is not implemented")


class _Cv2DestroyFailure:
    @staticmethod
    def destroyAllWindows():
        raise RuntimeError("The function is not implemented")


def test_require_gui_support_raises_clear_error_for_headless_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(g1_viewer_app, "cv2", _Cv2ProbeFailure())

    with pytest.raises(RuntimeError, match="OpenCV GUI support is unavailable"):
        g1_viewer_app._require_gui_support()


def test_close_windows_safely_swallows_destroy_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(g1_viewer_app, "cv2", _Cv2DestroyFailure())

    g1_viewer_app._close_windows_safely()

    captured = capsys.readouterr()
    assert "skipped OpenCV window teardown" in captured.out


def test_draw_overlay_renders_trajectory_polyline() -> None:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    overlay = {
        "detector_backend": "stub",
        "detections": [],
        "trajectory_pixels": [[80, 80], [96, 80], [112, 96]],
    }

    canvas = g1_viewer_app._draw_overlay(frame, overlay, frame_id=3, source="unit_test")

    assert tuple(int(v) for v in canvas[80, 80]) != (0, 0, 0)
    assert tuple(int(v) for v in canvas[96, 112]) != (0, 0, 0)


def test_build_view_canvas_keeps_rgb_only_by_default() -> None:
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    overlay = {"detector_backend": "stub", "detections": [], "trajectory_pixels": []}

    canvas = g1_viewer_app._build_view_canvas(
        rgb,
        overlay,
        frame_id=1,
        source="unit_test",
    )

    assert canvas.shape == (32, 48, 3)


def test_build_view_canvas_appends_depth_panel_when_enabled() -> None:
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    depth = np.full((32, 48), 1.5, dtype=np.float32)
    overlay = {"detector_backend": "stub", "detections": [], "trajectory_pixels": []}

    canvas = g1_viewer_app._build_view_canvas(
        rgb,
        overlay,
        frame_id=1,
        source="unit_test",
        depth_image_m=depth,
        show_depth=True,
        depth_min_m=0.0,
        depth_max_m=5.0,
    )

    assert canvas.shape == (32, 96, 3)
    assert np.any(canvas[:, 48:, :] != 0)
