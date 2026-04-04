from __future__ import annotations

import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apps.system2_app import create_app


def _jpeg_bytes() -> io.BytesIO:
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def _depth_png_bytes() -> io.BytesIO:
    image = Image.fromarray(np.ones((16, 16), dtype=np.uint16), mode="I;16")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class _FakeService:
    def __init__(self) -> None:
        self.ready = True

    def health_payload(self) -> dict[str, object]:
        return {"status": "ok", "ready": self.ready, "service": "system2_wrapper"}

    def eval_dual(self, image_file, depth_file, payload: dict[str, object]) -> dict[str, object]:  # noqa: ANN001
        _ = image_file, depth_file
        return {"pixel_goal": [12, 34], "echo": payload}

    def open_chat_session(self, payload: dict[str, object]) -> dict[str, object]:
        return {"status": "ok", "opened": payload.get("session_id", "chat-1")}

    def chat_session_message(self, payload: dict[str, object]) -> dict[str, object]:
        return {"status": "ok", "message": payload.get("message", "")}

    def close_chat_session(self, payload: dict[str, object]) -> dict[str, object]:
        return {"status": "ok", "closed": payload.get("session_id", "")}


def test_system2_app_routes_health_navigation_and_chat() -> None:
    service = _FakeService()
    app = create_app(service=service)
    client = app.test_client()

    health = client.get("/healthz")
    navigation = client.post(
        "/navigation",
        data={
            "json": '{"session_id":"nav-1","instruction":"dock","reset":true,"idx":0}',
            "image": (_jpeg_bytes(), "rgb.jpg"),
            "depth": (_depth_png_bytes(), "depth.png"),
        },
        content_type="multipart/form-data",
    )
    legacy = client.post(
        "/eval_dual",
        data={
            "json": '{"session_id":"nav-1","instruction":"dock","reset":false,"idx":1}',
            "image": (_jpeg_bytes(), "rgb.jpg"),
            "depth": (_depth_png_bytes(), "depth.png"),
        },
        content_type="multipart/form-data",
    )
    opened = client.post("/chat/session/open", json={"session_id": "chat-1"})
    replied = client.post("/chat/session/message", json={"session_id": "chat-1", "message": "hi"})
    closed = client.post("/chat/session/close", json={"session_id": "chat-1"})

    assert health.status_code == 200
    assert navigation.status_code == 200
    assert navigation.get_json()["pixel_goal"] == [12, 34]
    assert legacy.status_code == 200
    assert legacy.get_json()["echo"]["idx"] == 1
    assert opened.get_json()["opened"] == "chat-1"
    assert replied.get_json()["message"] == "hi"
    assert closed.get_json()["closed"] == "chat-1"


def test_system2_app_returns_503_when_sidecar_not_ready() -> None:
    service = _FakeService()
    service.ready = False
    app = create_app(service=service)
    client = app.test_client()

    response = client.get("/healthz")

    assert response.status_code == 503
