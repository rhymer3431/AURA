from __future__ import annotations

from dataclasses import dataclass

from ipc.messages import TaskRequest
from schemas.execution_mode import ExecutionMode
from services.intent_service import IntentService


_MEMORY_TOKENS = ("봤던", "아까", "previously", "remembered", "earlier", "이전에 본", "지나쳤던")
_EXPLORE_TOKENS = ("탐색", "둘러", "roam", "wander", "explore", "scan around", "search around", "local search")
_NAV_TOKENS = ("가", "이동", "navigate", "move", "go to", "reach", "follow", "찾")
_EMBODIED_NAV_TOKENS = (
    "go to",
    "move to",
    "navigate to",
    "head to",
    "inspect",
    "check",
    "look for",
    "find",
    "fetch",
    "bring",
    "가서",
    "가줘",
    "로 가",
    "으로 가",
    "쪽으로 가",
    "이동해",
    "이동해줘",
    "이동해 줘",
    "확인해",
    "확인해줘",
    "확인해 줘",
    "찾아",
    "찾아줘",
    "찾아 줘",
    "이동",
)
_TALK_TOKENS = ("왜", "무엇", "설명", "explain", "tell me", "what", "why", "how")


@dataclass(frozen=True)
class ModeClassification:
    mode: ExecutionMode
    reason: str
    target_class: str = ""
    intent_name: str = ""


class ExecutionModeClassifier:
    def __init__(self) -> None:
        self._intent_service = IntentService()

    def classify(self, request: TaskRequest) -> ModeClassification:
        instruction = str(request.command_text).strip()
        lowered = instruction.lower()
        target_json = dict(request.target_json or {})
        target_mode = str(target_json.get("target_mode", "")).strip().lower()
        pose_source = str(target_json.get("pose_source", "")).strip().lower()
        parsed = self._intent_service.parse(instruction, target_json=target_json)

        if pose_source == "memory" or any(token in instruction for token in _MEMORY_TOKENS):
            return ModeClassification(mode="MEM_NAV", reason="memory_cue", target_class=parsed.target_class, intent_name=parsed.name)

        if any(token in lowered for token in _EXPLORE_TOKENS) or target_mode == "explore":
            return ModeClassification(mode="EXPLORE", reason="explicit_explore", target_class=parsed.target_class, intent_name=parsed.name)

        if target_mode in {"follow_person", "goto_visible_object"} or parsed.name in {"follow", "goto_visible_object"}:
            return ModeClassification(mode="NAV", reason="live_navigation", target_class=parsed.target_class, intent_name=parsed.name)

        if parsed.target_class != "" and any(token in lowered for token in _NAV_TOKENS):
            return ModeClassification(mode="NAV", reason="target_navigation", target_class=parsed.target_class, intent_name=parsed.name)

        if any(token in lowered for token in _EMBODIED_NAV_TOKENS):
            return ModeClassification(mode="NAV", reason="embodied_instruction", target_class=parsed.target_class, intent_name=parsed.name)

        if any(token in lowered for token in _TALK_TOKENS):
            return ModeClassification(mode="TALK", reason="conversation", target_class=parsed.target_class, intent_name=parsed.name)

        if parsed.target_class != "":
            return ModeClassification(mode="NAV", reason="target_present", target_class=parsed.target_class, intent_name=parsed.name)

        return ModeClassification(mode="TALK", reason="default_talk", target_class=parsed.target_class, intent_name=parsed.name)
