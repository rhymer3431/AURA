from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Pose3D = tuple[float, float, float]


def pose3(values: tuple[float, float, float] | list[float] | Any) -> Pose3D:
    if values is None:
        return (0.0, 0.0, 0.0)
    seq = tuple(float(v) for v in values)
    if len(seq) == 2:
        return (seq[0], seq[1], 0.0)
    if len(seq) < 3:
        raise ValueError(f"Pose requires at least 2 coordinates, got {values!r}")
    return (seq[0], seq[1], seq[2])


@dataclass
class PlaceNode:
    place_id: str
    pose: Pose3D
    room_id: str = ""
    neighbors: list[str] = field(default_factory=list)
    visit_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectSnapshot:
    timestamp: float
    pose: Pose3D
    confidence: float
    note: str = ""


@dataclass
class ObjectNode:
    object_id: str
    class_name: str
    track_id: str
    last_pose: Pose3D
    last_place_id: str
    first_seen: float
    last_seen: float
    confidence: float
    movable: bool = True
    state: str = "unknown"
    embedding_id: str = ""
    snapshots: list[ObjectSnapshot] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    conflict_flag: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationEdge:
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 1.0
    last_updated: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObsObject:
    class_name: str
    pose: Pose3D
    timestamp: float
    confidence: float = 1.0
    track_id: str = ""
    place_id: str = ""
    room_id: str = ""
    movable: bool = True
    state: str = "visible"
    embedding_id: str = ""
    snapshots: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalEvent:
    event_type: str
    timestamp: float
    track_id: str = ""
    person_id: str = ""
    object_id: str = ""
    pose: Pose3D | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    episode_id: str
    command_text: str
    intent: str
    target_json: dict[str, Any]
    visited_places: list[str] = field(default_factory=list)
    objects_seen: list[str] = field(default_factory=list)
    candidate_object_ids: list[str] = field(default_factory=list)
    candidate_place_ids: list[str] = field(default_factory=list)
    semantic_rules_applied: list[str] = field(default_factory=list)
    follow_target_id: str = ""
    speaker_person_id: str = ""
    summary_tags: list[str] = field(default_factory=list)
    success: bool | None = None
    failure_reason: str = ""
    recovery_actions: list[str] = field(default_factory=list)
    policy_events: list[dict[str, Any]] = field(default_factory=list)
    summary_text: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0


@dataclass
class SemanticRule:
    rule_key: str
    description: str
    trigger_signature: str = ""
    rule_type: str = "heuristic"
    planner_hint: dict[str, Any] = field(default_factory=dict)
    support_count: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    last_updated: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemoryCandidate:
    candidate_id: str
    candidate_type: str
    object_id: str = ""
    place_id: str = ""
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemorySnapshot:
    query_text: str
    candidates: list[WorkingMemoryCandidate]
    selected_ids: list[str]


@dataclass
class RecallQuery:
    query_text: str
    target_class: str = ""
    intent: str = ""
    room_id: str = ""
    top_k: int = 3


@dataclass
class RecallResult:
    query: RecallQuery
    candidates: list[WorkingMemoryCandidate]
    semantic_rules: list[SemanticRule] = field(default_factory=list)
    selected_object: ObjectNode | None = None
    selected_place: PlaceNode | None = None


@dataclass(frozen=True)
class ScratchpadState:
    instruction: str = ""
    planner_mode: str = ""
    task_state: str = "idle"
    task_id: str = ""
    command_id: int = -1
    goal_summary: str = ""
    checked_locations: list[str] = field(default_factory=list)
    recent_hint: str = ""
    next_priority: str = ""
    updated_at: float = 0.0


@dataclass(frozen=True)
class KeyframeRecord:
    keyframe_id: str
    image_path: str
    crop_paths: list[str] = field(default_factory=list)
    summary: str = ""
    timestamp: float = 0.0
    source_frame_id: int = -1
    robot_pose: Pose3D = (0.0, 0.0, 0.0)
    robot_yaw_rad: float = 0.0
    room_id: str = ""
    focus_labels: list[str] = field(default_factory=list)
    focus_object_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedMemoryLine:
    text: str
    score: float
    source_type: str
    entity_id: str = ""
    keyframe_id: str = ""


@dataclass(frozen=True)
class MemoryContextBundle:
    instruction: str
    scratchpad: ScratchpadState | None = None
    text_lines: list[RetrievedMemoryLine] = field(default_factory=list)
    keyframes: list[KeyframeRecord] = field(default_factory=list)
    crop_path: str = ""
    latent_backend_hint: str = "llama.cpp_s2_only"


@dataclass
class AssociationResult:
    object_node: ObjectNode
    place_node: PlaceNode
    matched_existing: bool
    conflict_flag: bool
    score: float


def scratchpad_state_to_dict(state: ScratchpadState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "instruction": str(state.instruction),
        "planner_mode": str(state.planner_mode),
        "task_state": str(state.task_state),
        "task_id": str(state.task_id),
        "command_id": int(state.command_id),
        "goal_summary": str(state.goal_summary),
        "checked_locations": [str(item) for item in state.checked_locations],
        "recent_hint": str(state.recent_hint),
        "next_priority": str(state.next_priority),
        "updated_at": float(state.updated_at),
    }


def scratchpad_state_from_dict(payload: dict[str, Any] | None) -> ScratchpadState | None:
    if not isinstance(payload, dict):
        return None
    return ScratchpadState(
        instruction=str(payload.get("instruction", "")),
        planner_mode=str(payload.get("planner_mode", "")),
        task_state=str(payload.get("task_state", "idle")),
        task_id=str(payload.get("task_id", "")),
        command_id=int(payload.get("command_id", -1)),
        goal_summary=str(payload.get("goal_summary", "")),
        checked_locations=[str(item) for item in payload.get("checked_locations", []) if str(item).strip() != ""],
        recent_hint=str(payload.get("recent_hint", "")),
        next_priority=str(payload.get("next_priority", "")),
        updated_at=float(payload.get("updated_at", 0.0)),
    )


def keyframe_record_to_dict(record: KeyframeRecord) -> dict[str, Any]:
    return {
        "keyframe_id": str(record.keyframe_id),
        "image_path": str(record.image_path),
        "crop_paths": [str(path) for path in record.crop_paths],
        "summary": str(record.summary),
        "timestamp": float(record.timestamp),
        "source_frame_id": int(record.source_frame_id),
        "robot_pose": [float(value) for value in record.robot_pose],
        "robot_yaw_rad": float(record.robot_yaw_rad),
        "room_id": str(record.room_id),
        "focus_labels": [str(label) for label in record.focus_labels],
        "focus_object_ids": [str(object_id) for object_id in record.focus_object_ids],
    }


def keyframe_record_from_dict(payload: dict[str, Any] | None) -> KeyframeRecord | None:
    if not isinstance(payload, dict):
        return None
    return KeyframeRecord(
        keyframe_id=str(payload.get("keyframe_id", "")),
        image_path=str(payload.get("image_path", "")),
        crop_paths=[str(path) for path in payload.get("crop_paths", []) if str(path).strip() != ""],
        summary=str(payload.get("summary", "")),
        timestamp=float(payload.get("timestamp", 0.0)),
        source_frame_id=int(payload.get("source_frame_id", -1)),
        robot_pose=pose3(payload.get("robot_pose", (0.0, 0.0, 0.0))),
        robot_yaw_rad=float(payload.get("robot_yaw_rad", 0.0)),
        room_id=str(payload.get("room_id", "")),
        focus_labels=[str(label) for label in payload.get("focus_labels", []) if str(label).strip() != ""],
        focus_object_ids=[str(object_id) for object_id in payload.get("focus_object_ids", []) if str(object_id).strip() != ""],
    )


def retrieved_memory_line_to_dict(line: RetrievedMemoryLine) -> dict[str, Any]:
    return {
        "text": str(line.text),
        "score": float(line.score),
        "source_type": str(line.source_type),
        "entity_id": str(line.entity_id),
        "keyframe_id": str(line.keyframe_id),
    }


def retrieved_memory_line_from_dict(payload: dict[str, Any] | None) -> RetrievedMemoryLine | None:
    if not isinstance(payload, dict):
        return None
    return RetrievedMemoryLine(
        text=str(payload.get("text", "")),
        score=float(payload.get("score", 0.0)),
        source_type=str(payload.get("source_type", "")),
        entity_id=str(payload.get("entity_id", "")),
        keyframe_id=str(payload.get("keyframe_id", "")),
    )


def memory_context_to_dict(bundle: MemoryContextBundle | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return {
        "instruction": str(bundle.instruction),
        "scratchpad": scratchpad_state_to_dict(bundle.scratchpad),
        "text_lines": [retrieved_memory_line_to_dict(line) for line in bundle.text_lines],
        "keyframes": [keyframe_record_to_dict(record) for record in bundle.keyframes],
        "crop_path": str(bundle.crop_path),
        "latent_backend_hint": str(bundle.latent_backend_hint),
    }


def memory_context_from_dict(payload: dict[str, Any] | None) -> MemoryContextBundle | None:
    if not isinstance(payload, dict):
        return None
    return MemoryContextBundle(
        instruction=str(payload.get("instruction", "")),
        scratchpad=scratchpad_state_from_dict(payload.get("scratchpad")),
        text_lines=[
            line
            for line in (retrieved_memory_line_from_dict(item) for item in payload.get("text_lines", []))
            if line is not None and line.text.strip() != ""
        ],
        keyframes=[
            record
            for record in (keyframe_record_from_dict(item) for item in payload.get("keyframes", []))
            if record is not None and record.keyframe_id.strip() != ""
        ],
        crop_path=str(payload.get("crop_path", "")),
        latent_backend_hint=str(payload.get("latent_backend_hint", "llama.cpp_s2_only")),
    )
