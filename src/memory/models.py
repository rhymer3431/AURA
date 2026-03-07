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


@dataclass
class AssociationResult:
    object_node: ObjectNode
    place_node: PlaceNode
    matched_existing: bool
    conflict_flag: bool
    score: float
