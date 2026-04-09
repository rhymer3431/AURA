from __future__ import annotations

from typing import Any, TypedDict


class TaskFrameTarget(TypedDict):
    object: str | None
    instance_hint: str | None
    location_hint: str | None


class TaskFrameQuery(TypedDict):
    query_type: str | None
    attribute: str | None
    operator: str | None
    expected_value: str | None


class TaskFrameConstraints(TypedDict):
    return_after_check: bool
    report_result: bool


class TaskFrameClarification(TypedDict):
    required: bool
    question_ko: str | None


class TaskFrame(TypedDict):
    intent: str
    target: TaskFrameTarget
    query: TaskFrameQuery
    constraints: TaskFrameConstraints
    clarification: TaskFrameClarification


class Subgoal(TypedDict):
    id: str
    type: str
    status: str
    succeed: bool
    input: dict[str, Any]
    output: dict[str, Any]
    attempts: int
    failure_reason: str | None
