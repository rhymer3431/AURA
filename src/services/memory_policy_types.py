from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memory.models import MemoryContextBundle, RecallResult


class MemoryPolicyLabel(str, Enum):
    ROUTE_DIRECT_VISION = "ROUTE_DIRECT_VISION"
    ROUTE_MEMORY_VISION = "ROUTE_MEMORY_VISION"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    WAIT = "WAIT"


@dataclass(frozen=True)
class MemoryPolicyContext:
    instruction: str
    target_class: str
    task_state: str
    current_pose: tuple[float, float, float] | None
    visible_target_now: bool
    memory_context: MemoryContextBundle | None
    recall_result: RecallResult | None
    semantic_rule_hints: list[str] = field(default_factory=list)
    candidate_count: int = 0
    top_score: float = 0.0
    score_gap: float = 0.0
    retrieval_confidence: float = 0.0
    ambiguity: bool = False

    def feature_snapshot(self) -> dict[str, Any]:
        return {
            "target_class": self.target_class,
            "task_state": self.task_state,
            "visible_target_now": bool(self.visible_target_now),
            "candidate_count": int(self.candidate_count),
            "top_score": round(float(self.top_score), 4),
            "score_gap": round(float(self.score_gap), 4),
            "retrieval_confidence": round(float(self.retrieval_confidence), 4),
            "ambiguity": bool(self.ambiguity),
        }


@dataclass(frozen=True)
class MemoryPolicyDecision:
    label: MemoryPolicyLabel
    confidence: float
    source: str
    fallback_used: bool
    shadow_only: bool
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
