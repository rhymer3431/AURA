from __future__ import annotations

import io
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from server.system2_service import DEFAULT_PORT, System2Service


def _image_file(fill: int = 64) -> io.BytesIO:
    image = Image.fromarray(np.full((32, 32, 3), fill, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return buffer


def _depth_file(value_mm: int = 1500) -> io.BytesIO:
    depth = Image.fromarray(np.full((32, 32), value_mm, dtype=np.uint16), mode="I;16")
    buffer = io.BytesIO()
    depth.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class _FakeSidecar:
    def __init__(self, responses: list[dict[str, object]] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, object]] = []
        self.chat_lora_path = ROOT / "artifacts" / "models" / "chat-lora.gguf"

    def healthy(self, **_kwargs) -> bool:
        return True

    def health_payload(self) -> dict[str, object]:
        return {"llama_sidecar_ready": True}

    def close(self) -> None:
        return None

    def chat_lora_request(self) -> list[dict[str, object]]:
        return [{"id": 0, "scale": 1.0}]

    def chat_completion(self, *, messages, max_tokens, lora=None):  # noqa: ANN001
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "lora": lora,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return {"choices": [{"text": "STOP"}]}


def _args() -> Namespace:
    return Namespace(
        host="127.0.0.1",
        port=DEFAULT_PORT,
        resize_w=384,
        resize_h=384,
        num_history=4,
        plan_step_gap=4,
        max_new_tokens=16,
        prompt_model_path=None,
        llama_cpp_root=ROOT / "llama.cpp",
        llama_server_path=ROOT / "llama.cpp" / "llama-server.exe",
        llama_model_path=ROOT / "artifacts" / "models" / "InternVLA-N1-System2.Q4_K_M.gguf",
        llama_mmproj_path=ROOT / "artifacts" / "models" / "InternVLA-N1-System2.mmproj-Q8_0.gguf",
        llama_url="http://127.0.0.1:15802",
        llama_ctx_size=8192,
        llama_threads=None,
        llama_gpu_layers="all",
        llama_main_gpu=0,
        llama_flash_attn="on",
        llama_cache_type_k="q8_0",
        llama_cache_type_v="q8_0",
        llama_cache_prompt="0",
        llama_chat_lora_path=ROOT / "artifacts" / "models" / "InternVLA-Humanoid-Chat-LoRA.F16.gguf",
        llama_chat_lora_scale=1.0,
        chat_session_system_prompt="chat prompt",
        llama_health_timeout_s=30.0,
        llama_stdout_log=ROOT / "tmp" / "process_logs" / "system" / "internvla_llama.stdout.log",
        llama_stderr_log=ROOT / "tmp" / "process_logs" / "system" / "internvla_llama.stderr.log",
    )


def test_eval_dual_reuses_cached_pixel_goal_within_plan_step_gap() -> None:
    fake_sidecar = _FakeSidecar(
        responses=[
            {"choices": [{"text": "120, 80"}]},
        ]
    )
    service = System2Service(_args(), sidecar=fake_sidecar)

    first = service.eval_dual(
        _image_file(32),
        _depth_file(),
        {
            "session_id": "nav-1",
            "instruction": "dock",
            "idx": 1,
            "reset": True,
        },
    )
    second = service.eval_dual(
        _image_file(48),
        _depth_file(),
        {
            "session_id": "nav-1",
            "instruction": "dock",
            "idx": 2,
            "reset": False,
        },
    )

    assert first["pixel_goal"] == [80, 120]
    assert second["pixel_goal"] == [80, 120]
    assert len(fake_sidecar.calls) == 1
    assert fake_sidecar.calls[0]["lora"] is None


def test_eval_dual_retries_once_after_look_down() -> None:
    fake_sidecar = _FakeSidecar(
        responses=[
            {"choices": [{"text": "look down"}]},
            {"choices": [{"text": "200, 140"}]},
        ]
    )
    service = System2Service(_args(), sidecar=fake_sidecar)

    result = service.eval_dual(
        _image_file(96),
        _depth_file(),
        {
            "session_id": "nav-2",
            "instruction": "dock",
            "idx": 1,
            "reset": True,
        },
    )

    assert result["pixel_goal"] == [140, 200]
    assert len(fake_sidecar.calls) == 2
    health = service.health_payload()
    assert health["system2_output"]["decisionMode"] == "pixel_goal"
    assert health["system2_output"]["historyFrameIds"] == []


def test_chat_sessions_apply_lora_and_navigation_does_not() -> None:
    fake_sidecar = _FakeSidecar(
        responses=[
            {"choices": [{"text": "121, 84"}]},
            {"choices": [{"text": "hello back"}]},
        ]
    )
    service = System2Service(_args(), sidecar=fake_sidecar)

    nav = service.eval_dual(
        _image_file(64),
        _depth_file(),
        {
            "session_id": "nav-3",
            "instruction": "dock",
            "idx": 1,
            "reset": True,
        },
    )
    opened = service.open_chat_session({"session_id": "chat-1"})
    replied = service.chat_session_message({"session_id": "chat-1", "message": "hi", "max_tokens": 32})
    closed = service.close_chat_session({"session_id": "chat-1"})

    assert nav["pixel_goal"] == [84, 121]
    assert opened["chat_lora_active"] is True
    assert replied["message"]["content"] == "hello back"
    assert closed["closed"] is True
    assert fake_sidecar.calls[0]["lora"] is None
    assert fake_sidecar.calls[1]["lora"] == [{"id": 0, "scale": 1.0}]


def test_health_payload_exposes_external_defaults() -> None:
    service = System2Service(_args(), sidecar=_FakeSidecar())

    payload = service.health_payload()

    assert payload["status"] == "ok"
    assert payload["ready"] is True
    assert payload["port"] == 15801
    assert payload["navigation_route"] == "/navigation"
