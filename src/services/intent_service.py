from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_TARGET_ALIASES = {
    "사과": "apple",
    "apple": "apple",
    "사람": "person",
    "person": "person",
    "빨간 큐브": "cube",
    "red cube": "cube",
    "cube": "cube",
}


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
        target_class = self._extract_target_class(lowered, target_json)
        target_track_id = str(target_json.get("target_track_id", "")).strip()

        if any(token in lowered for token in ("따라와", "따라", "follow")):
            return ParsedIntent(name="follow", target_class="person", target_track_id=target_track_id)
        if any(token in lowered for token in ("부르면", "caller", "look at", "call")):
            return ParsedIntent(name="attend_caller", metadata={"source": "speaker_event"})
        if any(token in lowered for token in ("아까", "remembered", "봤던", "find", "찾아가", "찾아")) and target_class != "":
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
