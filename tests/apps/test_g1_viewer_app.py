from __future__ import annotations

import sys
from pathlib import Path

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
