from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import inference.vlm.system2_session as system2_module
from inference.vlm import System2Session, System2SessionConfig, parse_system2_output


def test_parse_system2_output_handles_pixel_goal() -> None:
    decision = parse_system2_output("321, 123", width=640, height=480)
    assert decision.mode == "pixel_goal"
    assert decision.pixel_goal == (123, 321)


def test_build_vlm_endpoint_points_to_navigation() -> None:
    assert system2_module.build_vlm_endpoint("http://127.0.0.1:15801") == "http://127.0.0.1:15801/navigation"
    assert system2_module.build_vlm_endpoint("http://127.0.0.1:15801/navigation") == "http://127.0.0.1:15801/navigation"


def test_system2_session_reset_and_prepare_request() -> None:
    session = System2Session(System2SessionConfig(endpoint="http://127.0.0.1:15801"))
    session.reset("dock")
    request = session.prepare_request(frame_id=7, width=640, height=480)

    assert request.body["reset"] is True
    assert request.body["idx"] == 0
    assert request.body["instruction"] == "dock"


def test_system2_session_step_posts_multipart(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"pixel_goal": [80, 120]}

    def _fake_post(url, *, files, data, timeout):  # noqa: ANN001
        captured["url"] = url
        captured["files"] = files
        captured["data"] = data
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(system2_module.requests, "post", _fake_post)
    session = System2Session(System2SessionConfig(endpoint="http://127.0.0.1:15801", timeout_sec=9.0))
    session.reset("dock")
    result = session.step_session(
        session_id="nav-1",
        rgb=np.zeros((32, 32, 3), dtype=np.uint8),
        depth=np.ones((32, 32), dtype=np.float32),
        stamp_s=1.5,
    )

    payload = json.loads(captured["data"]["json"])
    assert captured["url"] == "http://127.0.0.1:15801/navigation"
    assert set(captured["files"].keys()) == {"image", "depth"}
    assert payload["reset"] is True
    assert payload["instruction"] == "dock"
    assert result.decision_mode == "pixel_goal"
    assert tuple(result.pixel_xy.tolist()) == (31.0, 31.0)


def test_sample_depth_window_and_projection() -> None:
    depth = np.ones((10, 10), dtype=np.float32)
    intrinsic = np.asarray([[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    world_xy = system2_module.resolve_goal_world_xy_from_pixel(
        pixel_xy=(5, 5),
        depth_image=depth,
        intrinsic=intrinsic,
        camera_pos_world=np.zeros(3, dtype=np.float32),
        camera_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert world_xy is not None
    assert len(world_xy) == 2
