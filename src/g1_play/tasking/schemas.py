from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class RobotState(TypedDict):
    current_room: str | None
    holding_object: str | None


class WorldSummary(TypedDict):
    known_rooms: list[str]
    recent_seen: list[dict[str, Any]]


class Capabilities(TypedDict):
    detectable_objects: list[str]
    inspectable_attributes: dict[str, list[str]]
    can_return_home: bool


class PlanRequest(TypedDict):
    utterance_ko: str
    robot_state: RobotState
    world_summary: WorldSummary
    capabilities: Capabilities


class PlanHints(TypedDict):
    room: str | None
    instance: str | None


class PlanConstraints(TypedDict):
    return_home: bool
    report_result: bool


class PlanClarification(TypedDict):
    required: bool
    question_ko: str | None


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


class PlanResponse(TypedDict):
    intent: str
    plan_template: str
    target: dict[str, Any]
    hints: PlanHints
    constraints: PlanConstraints
    clarification: PlanClarification


class RepairRequest(TypedDict):
    failure_type: str
    target_class: str | None
    searched_rooms: list[str]
    remaining_rooms: list[str]
    recent_seen: dict[str, Any] | None
    retries: int


class RepairClarification(TypedDict):
    required: bool
    question_ko: str | None


class RepairResponse(TypedDict):
    repair_template: str
    clarification: RepairClarification


class Subgoal(TypedDict):
    id: str
    type: str
    status: str
    succeed: bool
    input: dict[str, Any]
    output: dict[str, Any]
    attempts: int
    failure_reason: str | None


@dataclass(frozen=True)
class CompiledNode:
    type: str
    target: str | None = None
    attribute: str | None = None
    room_hint: str | None = None
    instance_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledMission:
    intent: str
    plan_template: str
    target_class: str | None
    attribute: str | None
    hints: dict[str, str | None]
    constraints: dict[str, bool]
    nodes: list[CompiledNode]


@dataclass(frozen=True)
class CompiledRepair:
    repair_template: str
    target_class: str | None
    nodes: list[CompiledNode]


@dataclass(frozen=True)
class CompletionDecision:
    done: bool
    success: bool
    reason: str | None = None
    retryable: bool = False
