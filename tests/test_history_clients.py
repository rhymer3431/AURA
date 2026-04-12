from __future__ import annotations

import numpy as np

from systems.inference.client import InternVlaNavClient
from systems.memory.api.runtime import decode_rgb_history_npz
from systems.navigation.client import NavDpClient


def _rgb(frame_id: int) -> np.ndarray:
    return np.full((3, 4, 3), frame_id, dtype=np.uint8)


def _depth() -> np.ndarray:
    return np.full((3, 4), 1.0, dtype=np.float32)


def test_system2_client_serializes_optional_history_npz() -> None:
    client = InternVlaNavClient(server_url="http://unused", timeout_s=1.0)
    captured: dict[str, object] = {}

    def _fake_post(path: str, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return {"discrete_action": [5]}

    client._post = _fake_post  # type: ignore[method-assign]
    client.reset_session(
        session_id="session-1",
        instruction="go to chair",
        language="en",
        image_width=4,
        image_height=3,
    )

    history = np.stack((_rgb(1), _rgb(2)), axis=0)
    client.step_session(
        session_id="session-1",
        rgb=_rgb(3),
        depth=_depth(),
        stamp_s=1.0,
        rgb_history=history,
    )

    files = captured["kwargs"]["files"]  # type: ignore[index]
    assert "history_npz" in files
    _, payload, content_type = files["history_npz"]
    assert content_type == "application/octet-stream"
    decoded = decode_rgb_history_npz(payload)
    assert np.array_equal(decoded, history)


def test_navdp_client_serializes_optional_history_npz() -> None:
    client = NavDpClient(server_url="http://unused", timeout_s=1.0)
    captured: dict[str, object] = {}

    def _fake_post(path: str, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return {"trajectory": [[0.0, 0.0, 0.0]], "all_trajectory": None, "all_values": None}

    client._post = _fake_post  # type: ignore[method-assign]

    history = np.stack((_rgb(4), _rgb(5)), axis=0)
    client.step_pointgoal(
        np.asarray((1.0, 0.0), dtype=np.float32),
        _rgb(6),
        _depth(),
        rgb_history=history,
    )

    files = captured["kwargs"]["files"]  # type: ignore[index]
    assert "history_npz" in files
    _, payload, content_type = files["history_npz"]
    assert content_type == "application/octet-stream"
    decoded = decode_rgb_history_npz(payload)
    assert np.array_equal(decoded, history)


def test_clients_omit_history_file_when_no_history_is_supplied() -> None:
    system2 = InternVlaNavClient(server_url="http://unused", timeout_s=1.0)
    navdp = NavDpClient(server_url="http://unused", timeout_s=1.0)
    system2_captured: dict[str, object] = {}
    navdp_captured: dict[str, object] = {}

    def _fake_system2_post(path: str, **kwargs):
        system2_captured["kwargs"] = kwargs
        return {"discrete_action": [5]}

    def _fake_navdp_post(path: str, **kwargs):
        navdp_captured["kwargs"] = kwargs
        return {"trajectory": [[0.0, 0.0, 0.0]], "all_trajectory": None, "all_values": None}

    system2._post = _fake_system2_post  # type: ignore[method-assign]
    navdp._post = _fake_navdp_post  # type: ignore[method-assign]

    system2.reset_session(
        session_id="session-2",
        instruction="go to table",
        language="en",
        image_width=4,
        image_height=3,
    )
    system2.step_session(session_id="session-2", rgb=_rgb(1), depth=_depth(), stamp_s=1.0)
    navdp.step_pointgoal(np.asarray((0.0, 1.0), dtype=np.float32), _rgb(2), _depth())

    assert "history_npz" not in system2_captured["kwargs"]["files"]  # type: ignore[index]
    assert "history_npz" not in navdp_captured["kwargs"]["files"]  # type: ignore[index]
