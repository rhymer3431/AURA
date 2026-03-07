from __future__ import annotations

import json
from typing import Any

from .messages import MessagePayload, message_from_dict, message_to_dict


def encode_message(message: MessagePayload) -> bytes:
    return json.dumps(message_to_dict(message), ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def decode_message(payload: bytes | bytearray | memoryview | str) -> MessagePayload:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = bytes(payload)
    return message_from_dict(json.loads(raw.decode("utf-8")))


def encode_envelope(topic: str, message: MessagePayload) -> tuple[bytes, bytes]:
    return topic.encode("utf-8"), encode_message(message)


def decode_envelope(topic: bytes | str, payload: bytes | str) -> tuple[str, MessagePayload]:
    topic_text = topic.decode("utf-8") if isinstance(topic, bytes) else str(topic)
    return topic_text, decode_message(payload)


def encode_json(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
