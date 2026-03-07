from __future__ import annotations

import numpy as np

from adapters.navdp_http import NavDPClient, NavDPClientConfig


class _DummyResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._body


def test_nogoal_step_posts_rgbd_and_parses_trajectory(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, *, files, data=None, timeout):  # noqa: ANN001
        captured["url"] = url
        captured["files"] = files
        captured["data"] = data
        captured["timeout"] = timeout
        return _DummyResponse(
            {
                "trajectory": [[0.1, 0.0, 0.0], [0.2, 0.1, 0.0]],
                "all_trajectory": [[[0.1, 0.0, 0.0], [0.2, 0.1, 0.0]]],
                "all_values": [[1.0, 0.5]],
            }
        )

    monkeypatch.setattr("adapters.navdp_http.requests.post", fake_post)

    client = NavDPClient(NavDPClientConfig(base_url="http://127.0.0.1:8888", timeout_sec=2.5))
    response = client.nogoal_step(
        rgb_images=np.zeros((1, 4, 4, 3), dtype=np.uint8),
        depth_images_m=np.ones((1, 4, 4), dtype=np.float32),
    )

    assert str(captured["url"]).endswith("/nogoal_step")
    assert captured["data"] is None
    assert "image" in captured["files"]
    assert "depth" in captured["files"]
    assert float(captured["timeout"]) == 2.5
    assert response.trajectory.shape == (2, 3)
    assert response.all_trajectory.shape == (1, 2, 3)
    assert response.all_values.shape == (1, 2)
