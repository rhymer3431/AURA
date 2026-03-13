from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apps import frame_bridge_app


def test_live_mode_uses_standalone_branch(monkeypatch) -> None:
    called: dict[str, object] = {}

    monkeypatch.setattr(frame_bridge_app, "isaac_bootstrap_available", lambda: (True, ""))
    def _run_live(args) -> int:  # noqa: ANN001
        called["live_frame_source"] = str(args.frame_source)
        return 0

    monkeypatch.setattr(frame_bridge_app, "run_live_frame_bridge_app", _run_live)
    monkeypatch.setattr(
        frame_bridge_app,
        "run_lightweight_frame_bridge",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("lightweight bridge should not be selected")),
    )
    monkeypatch.setattr(
        frame_bridge_app,
        "build_frame_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("build_frame_source must not run before live bootstrap")),
    )

    assert frame_bridge_app.main(["--frame-source", "live"]) == 0
    assert called["live_frame_source"] == "live"


def test_auto_mode_falls_back_to_lightweight_bridge_when_isaac_unavailable(monkeypatch) -> None:
    called: dict[str, object] = {}

    monkeypatch.setattr(frame_bridge_app, "isaac_bootstrap_available", lambda: (False, "isaac unavailable"))
    monkeypatch.setattr(
        frame_bridge_app,
        "run_live_frame_bridge_app",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live bridge should not run when Isaac is unavailable")),
    )

    def _run_lightweight(args, *, force_synthetic: bool = False, initial_notice: str = "") -> int:
        called["force_synthetic"] = force_synthetic
        called["initial_notice"] = initial_notice
        called["frame_source"] = str(args.frame_source)
        return 0

    monkeypatch.setattr(frame_bridge_app, "run_lightweight_frame_bridge", _run_lightweight)

    assert frame_bridge_app.main(["--frame-source", "auto"]) == 0
    assert called["force_synthetic"] is True
    assert called["frame_source"] == "auto"
    assert called["initial_notice"] == "isaac unavailable"
