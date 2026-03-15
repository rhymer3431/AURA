from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_TARGET_ALIASES = {
    "사과": "apple",
    "apple": "apple",
    "상자": "box",
    "box": "box",
    "파란 상자": "box",
    "blue box": "box",
    "의자": "chair",
    "chair": "chair",
    "초록 의자": "chair",
    "green chair": "chair",
    "병": "bottle",
    "bottle": "bottle",
    "오렌지 병": "bottle",
    "orange bottle": "bottle",
    "문": "door",
    "door": "door",
    "빨간 문": "door",
    "red door": "door",
    "사람": "person",
    "person": "person",
    "빨간 큐브": "cube",
    "red cube": "cube",
    "cube": "cube",
}

_MEMORY_TRIGGER_TOKENS = (
    "아까",
    "이전에 본",
    "전에 본",
    "봤던",
    "지나쳤던",
    "remembered",
    "earlier",
    "previously",
    "다시 찾아",
    "다시 가",
    "다시 찾아가",
    "find again",
    "찾아가",
    "찾아",
)


@dataclass(frozen=True)
class ParsedIntent:
    name: str
    target_class: str = ""
    target_track_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class IntentService:
    def parse(self, command_text: str, *, target_json: dict[str, Any] | None = None) -> ParsedIntent:
        text = str(command_text).strip()
        lowered = text.lower()
        target_json = dict(target_json or {})
        target_mode = str(target_json.get("target_mode", "")).strip().lower()
        target_class = self._extract_target_class(lowered, target_json)
        target_track_id = str(target_json.get("target_track_id", "")).strip()

        if target_mode == "follow_person":
            return ParsedIntent(name="follow", target_class="person", target_track_id=target_track_id)
        if target_mode == "goto_visible_object":
            return ParsedIntent(name="goto_visible_object", target_class=target_class, target_track_id=target_track_id)

        if any(token in lowered for token in ("따라와", "따라", "follow")):
            return ParsedIntent(name="follow", target_class="person", target_track_id=target_track_id)
        if any(token in lowered for token in ("부르면", "caller", "look at", "call")):
            return ParsedIntent(name="attend_caller", metadata={"source": "speaker_event"})
        if any(token in lowered for token in _MEMORY_TRIGGER_TOKENS) and target_class != "":
            return ParsedIntent(name="goto_remembered_object", target_class=target_class)
        return ParsedIntent(name="local_search", target_class=target_class)

    def _extract_target_class(self, lowered_text: str, target_json: dict[str, Any]) -> str:
        explicit = str(target_json.get("target_class", "")).strip().lower()
        if explicit != "":
            return explicit
        for alias, normalized in _TARGET_ALIASES.items():
            if alias in lowered_text:
                return normalized
        return ""
