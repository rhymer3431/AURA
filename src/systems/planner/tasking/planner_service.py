from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from systems.planner.tasking.normalizer import (
    detect_attribute,
    detect_desired_check,
    detect_instance_hint,
    detect_object_class,
    detect_room_hints,
    infer_intent,
    normalize_text,
)
from systems.planner.tasking.ontology import QUERY_OPERATORS, QUERY_TYPES, TASK_FRAME_INTENTS
from systems.planner.tasking.task_frames import expected_value_from_desired_check, task_frame_to_plan
from systems.planner.tasking.validator import (
    PlannerValidationError,
    validate_plan_request,
    validate_plan_response,
    validate_task_frame_response,
)
from systems.inference.api.planner import CompletionFn, PlannerClientError, call_json_with_retry

DEFAULT_MODEL = "Qwen3-1.7B-Q4_K_M-Instruct.gguf"

RETURN_HOME_KEYWORDS = (
    "\ub3cc\uc544\uc640",
    "\uac14\ub2e4 \uc640",
    "\ud655\uc778\ud558\uace0 \uc640",
    "\ub2e4\ub140\uc640",
    "\ubcf4\uace0 \uc640",
    "come back",
    "return",
    "and come back",
)
HERE_ROOM_KEYWORDS = (
    "\uc5ec\uae30",
    "\uc774 \ubc29",
    "\ud604\uc7ac \ubc29",
    "here",
    "this room",
    "current room",
)
COMMAND_KEYWORDS = (
    "\ud655\uc778",
    "\ubd10\uc918",
    "\ucc3e\uc544",
    "\uc774\ub3d9",
    "\uac00\uc918",
    "\ub2e4\uac00\uac00",
    "\uc0c1\ud0dc",
    "check",
    "inspect",
    "find",
    "navigate",
    "go to",
    "move to",
    "status",
)

CLARIFICATION_MESSAGES = {
    "multiple_explicit_rooms": "\uc5b4\ub290 \ubc29\uc758 \ub300\uc0c1\uc744 \ub9d0\ud558\ub294\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
    "multiple_recent_target_rooms": "\uac19\uc740 \ub300\uc0c1\uc774 \uc5ec\ub7ec \uacf3\uc5d0 \uc788\uc2b5\ub2c8\ub2e4. \uc5b4\ub290 \ubc29\uc758 \ub300\uc0c1\uc778\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
    "missing_target": "\uc5b4\ub5a4 \ub300\uc0c1\uc744 \ud655\uc778\ud574\uc57c \ud558\ub294\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
    "missing_attribute": "\uc5b4\ub5a4 \uc18d\uc131\uc744 \ud655\uc778\ud574\uc57c \ud558\ub294\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
}


def _dump_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


@dataclass(frozen=True)
class PlanSemanticAnalysis:
    normalized_utterance: str
    intent_candidate: str
    target_class_candidate: str | None
    attribute_candidate: str | None
    expected_value_candidate: str | None
    explicit_room_candidates: tuple[str, ...]
    recent_target_rooms: tuple[str, ...]
    preferred_room: str | None
    instance_hint: str | None
    return_home_requested: bool
    clarification_reasons: tuple[str, ...]
    unsupported_reasons: tuple[str, ...]


def build_task_frame_messages(request: dict[str, Any], analysis: PlanSemanticAnalysis) -> list[dict[str, str]]:
    semantic_summary = {
        "intent_candidate": analysis.intent_candidate,
        "target_class_candidate": analysis.target_class_candidate,
        "attribute_candidate": analysis.attribute_candidate,
        "expected_value_candidate": analysis.expected_value_candidate,
        "explicit_room_candidates": list(analysis.explicit_room_candidates),
        "recent_target_rooms": list(analysis.recent_target_rooms),
        "preferred_room": analysis.preferred_room,
        "instance_hint": analysis.instance_hint,
        "return_home_requested": analysis.return_home_requested,
        "clarification_reasons": list(analysis.clarification_reasons),
        "unsupported_reasons": list(analysis.unsupported_reasons),
    }
    system = f"""You are a semantic planner for a mobile robot.

Your job:
- Convert the user's Korean command into a fixed JSON task frame.
- Fill only the allowed enum values.
- Never generate free-form navigation instructions.
- Never invent rooms, objects, attributes, operators, or capabilities.
- Use the semantic summary as advisory normalized context.
- If the command is ambiguous and affects execution, set clarification.required=true.
- Output JSON only.

Allowed intents:
{", ".join(TASK_FRAME_INTENTS)}

Allowed query types:
{", ".join(QUERY_TYPES)}

Allowed query operators:
{", ".join(QUERY_OPERATORS)}

Required JSON schema:
{{
  "intent": "string",
  "target": {{
    "object": "string or null",
    "instance_hint": "string or null",
    "location_hint": "string or null"
  }},
  "query": {{
    "query_type": "string or null",
    "attribute": "string or null",
    "operator": "string or null",
    "expected_value": "string or null"
  }},
  "constraints": {{
    "return_after_check": false,
    "report_result": true
  }},
  "clarification": {{
    "required": false,
    "question_ko": null
  }}
}}
"""
    user = (
        "Structured input:\n"
        + _dump_json(request)
        + "\n\nSemantic summary:\n"
        + _dump_json(semantic_summary)
        + "\n\nReturn JSON only."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass
class PlannerService:
    completion: CompletionFn | None = None
    model: str = DEFAULT_MODEL
    timeout: float = 120.0
    temperature: float = 0.1
    max_tokens: int = 192

    def plan_task_frame(self, request: dict[str, Any]) -> dict[str, Any]:
        request = validate_plan_request(request)
        analysis = self._analyze_request(request)
        validator = lambda payload: validate_task_frame_response(payload, request["capabilities"])
        if self.completion is None:
            return validator(self._build_task_frame_from_analysis(request, analysis))
        try:
            return call_json_with_retry(
                self.completion,
                build_task_frame_messages(request, analysis),
                self.model,
                self.timeout,
                self.temperature,
                self.max_tokens,
                validator,
            )
        except (PlannerClientError, PlannerValidationError):
            return validator(self._build_task_frame_from_analysis(request, analysis))

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        request = validate_plan_request(request)
        task_frame = self.plan_task_frame(request)
        return validate_plan_response(task_frame_to_plan(task_frame), request["capabilities"])

    def _analyze_request(self, request: dict[str, Any]) -> PlanSemanticAnalysis:
        utterance = request["utterance_ko"]
        normalized_utterance = normalize_text(utterance)
        capabilities = request["capabilities"]
        current_room = request["robot_state"].get("current_room")
        target_class = detect_object_class(utterance)
        attribute = detect_attribute(utterance, target_class)
        desired_check = detect_desired_check(utterance, attribute)
        expected_value = expected_value_from_desired_check(desired_check)
        explicit_room_candidates = tuple(
            detect_room_hints(utterance, request["world_summary"]["known_rooms"])
        )
        recent_target_rooms = tuple(self._collect_recent_target_rooms(request, target_class))
        preferred_room = self._select_preferred_room(
            explicit_room_candidates,
            recent_target_rooms,
            current_room,
            normalized_utterance,
        )
        instance_hint = detect_instance_hint(utterance)
        return_home_requested = _contains_any(normalized_utterance, RETURN_HOME_KEYWORDS)
        intent = infer_intent(utterance, target_class, attribute)

        clarification_reasons: list[str] = []
        unsupported_reasons: list[str] = []

        if len(explicit_room_candidates) > 1:
            clarification_reasons.append("multiple_explicit_rooms")
        elif target_class is not None and not explicit_room_candidates and len(recent_target_rooms) > 1:
            clarification_reasons.append("multiple_recent_target_rooms")

        if target_class is None and _contains_any(normalized_utterance, COMMAND_KEYWORDS):
            clarification_reasons.append("missing_target")

        if intent == "inspect_attribute" and target_class is not None and attribute is None:
            inspectable = capabilities["inspectable_attributes"].get(target_class, [])
            if len(inspectable) > 1:
                clarification_reasons.append("missing_attribute")
            elif len(inspectable) == 1:
                attribute = inspectable[0]

        if target_class is not None and target_class not in capabilities["detectable_objects"]:
            unsupported_reasons.append("undetectable_target")

        if attribute is not None:
            inspectable = set(capabilities["inspectable_attributes"].get(target_class or "", ()))
            if attribute not in inspectable:
                unsupported_reasons.append("uninspectable_attribute")

        if return_home_requested and not capabilities["can_return_home"]:
            unsupported_reasons.append("cannot_return_home")

        return PlanSemanticAnalysis(
            normalized_utterance=normalized_utterance,
            intent_candidate=intent,
            target_class_candidate=target_class,
            attribute_candidate=attribute,
            expected_value_candidate=expected_value,
            explicit_room_candidates=explicit_room_candidates,
            recent_target_rooms=recent_target_rooms,
            preferred_room=preferred_room,
            instance_hint=instance_hint,
            return_home_requested=return_home_requested,
            clarification_reasons=tuple(_dedupe(clarification_reasons)),
            unsupported_reasons=tuple(_dedupe(unsupported_reasons)),
        )

    def _collect_recent_target_rooms(self, request: dict[str, Any], target_class: str | None) -> list[str]:
        if target_class is None:
            return []
        recent_seen = request["world_summary"]["recent_seen"]
        matching_rows: list[tuple[int | None, str]] = []
        for row in recent_seen:
            row_class = row.get("class")
            row_room = row.get("room")
            if row_class != target_class or not isinstance(row_room, str):
                continue
            age_value = row.get("age_sec")
            age_sec = age_value if isinstance(age_value, int) else None
            matching_rows.append((age_sec, row_room))
        matching_rows.sort(key=lambda item: item[0] if item[0] is not None else 10**9)
        return _dedupe([room for _, room in matching_rows])

    def _select_preferred_room(
        self,
        explicit_room_candidates: tuple[str, ...],
        recent_target_rooms: tuple[str, ...],
        current_room: str | None,
        normalized_utterance: str,
    ) -> str | None:
        if len(explicit_room_candidates) == 1:
            return explicit_room_candidates[0]
        if current_room and _contains_any(normalized_utterance, HERE_ROOM_KEYWORDS):
            return current_room
        if len(recent_target_rooms) == 1:
            return recent_target_rooms[0]
        return None

    def _build_task_frame_from_analysis(
        self,
        request: dict[str, Any],
        analysis: PlanSemanticAnalysis,
    ) -> dict[str, Any]:
        capabilities = request["capabilities"]
        intent = analysis.intent_candidate
        target_class = analysis.target_class_candidate
        attribute = analysis.attribute_candidate
        expected_value = analysis.expected_value_candidate
        room_hint = analysis.preferred_room
        instance_hint = analysis.instance_hint
        return_after_check = analysis.return_home_requested and bool(capabilities["can_return_home"])

        if analysis.clarification_reasons:
            reason = analysis.clarification_reasons[0]
            clarification_target_class = target_class
            clarification_attribute = attribute
            clarification_expected_value = expected_value
            if reason == "missing_target":
                clarification_target_class = None
                clarification_attribute = None
                clarification_expected_value = None
            return {
                "intent": "ask_clarification",
                "target": {
                    "object": clarification_target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": "attribute_check" if clarification_attribute is not None else None,
                    "attribute": clarification_attribute,
                    "operator": "equals" if clarification_attribute is not None else None,
                    "expected_value": clarification_expected_value,
                },
                "constraints": {"return_after_check": False, "report_result": True},
                "clarification": {
                    "required": True,
                    "question_ko": CLARIFICATION_MESSAGES[reason],
                },
            }

        if analysis.unsupported_reasons:
            query_attribute = attribute if target_class is not None else None
            return {
                "intent": "unsupported",
                "target": {
                    "object": target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": "attribute_check" if query_attribute is not None else None,
                    "attribute": query_attribute,
                    "operator": "equals" if query_attribute is not None else None,
                    "expected_value": expected_value if query_attribute is not None else None,
                },
                "constraints": {"return_after_check": False, "report_result": True},
                "clarification": {"required": False, "question_ko": None},
            }

        if intent == "inspect_attribute" and target_class is not None and attribute is not None:
            return {
                "intent": "check_state",
                "target": {
                    "object": target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": "attribute_check",
                    "attribute": attribute,
                    "operator": "equals",
                    "expected_value": expected_value,
                },
                "constraints": {"return_after_check": return_after_check, "report_result": True},
                "clarification": {"required": False, "question_ko": None},
            }

        if intent == "find_object" and target_class is not None:
            return {
                "intent": "find_object",
                "target": {
                    "object": target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": None,
                    "attribute": None,
                    "operator": None,
                    "expected_value": None,
                },
                "constraints": {"return_after_check": return_after_check, "report_result": True},
                "clarification": {"required": False, "question_ko": None},
            }

        if intent == "navigate_to_object" and target_class is not None:
            return {
                "intent": "navigate_to_object",
                "target": {
                    "object": target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": None,
                    "attribute": None,
                    "operator": None,
                    "expected_value": None,
                },
                "constraints": {"return_after_check": return_after_check, "report_result": True},
                "clarification": {"required": False, "question_ko": None},
            }

        query_attribute = attribute if target_class is not None else None
        return {
            "intent": "unsupported",
            "target": {
                "object": target_class,
                "instance_hint": instance_hint,
                "location_hint": room_hint,
            },
            "query": {
                "query_type": "attribute_check" if query_attribute is not None else None,
                "attribute": query_attribute,
                "operator": "equals" if query_attribute is not None else None,
                "expected_value": expected_value if query_attribute is not None else None,
            },
            "constraints": {"return_after_check": False, "report_result": True},
            "clarification": {"required": False, "question_ko": None},
        }
