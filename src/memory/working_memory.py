from __future__ import annotations

import math

from .association import xy_distance
from .models import ObjectNode, PlaceNode, SemanticRule, WorkingMemoryCandidate


class WorkingMemory:
    def select_candidates(
        self,
        *,
        query_text: str,
        objects: list[ObjectNode],
        places: dict[str, PlaceNode],
        semantic_rules: list[SemanticRule],
        current_pose: tuple[float, float, float] | None = None,
        target_class: str = "",
        room_id: str = "",
        top_k: int = 3,
    ) -> list[WorkingMemoryCandidate]:
        now_candidates: list[WorkingMemoryCandidate] = []
        semantic_strength = max((rule.success_rate for rule in semantic_rules), default=0.0)
        for obj in objects:
            place = places.get(obj.last_place_id)
            recency = math.exp(-max(0.0, float(objects[0].last_seen - obj.last_seen) if objects else 0.0) / 300.0)
            if current_pose is None or place is None:
                reachability = 0.5
            else:
                reachability = 1.0 / (1.0 + xy_distance(current_pose, place.pose))
            class_match = 1.0 if target_class != "" and obj.class_name.lower() == target_class.lower() else 0.5
            if target_class == "" and obj.class_name.lower() in query_text.lower():
                class_match = 1.0
            context_match = 1.0 if room_id != "" and place is not None and place.room_id == room_id else 0.5
            stale_penalty = min(max(0.0, (objects[0].last_seen - obj.last_seen) if objects else 0.0) / 1800.0, 1.0)
            semantic_rule_bonus = min(max(float(semantic_strength), 0.0), 1.0) * 0.15
            rule_context_bonus = self._rule_context_bonus(obj=obj, place=place, semantic_rules=semantic_rules)
            score = (
                0.30 * class_match
                + 0.20 * recency
                + 0.20 * reachability
                + 0.15 * float(obj.confidence)
                + 0.10 * context_match
                + semantic_rule_bonus
                + rule_context_bonus
                - 0.15 * stale_penalty
            )
            now_candidates.append(
                WorkingMemoryCandidate(
                    candidate_id=obj.object_id,
                    candidate_type="object",
                    object_id=obj.object_id,
                    place_id=obj.last_place_id,
                    score=float(score),
                    score_breakdown={
                        "semantic_match": float(semantic_strength),
                        "semantic_rule_bonus": float(semantic_rule_bonus),
                        "rule_context_bonus": float(rule_context_bonus),
                        "recency": float(recency),
                        "reachability": float(reachability),
                        "confidence": float(obj.confidence),
                        "context_match": float(context_match),
                        "stale_penalty": float(stale_penalty),
                    },
                    active=True,
                    metadata={"class_name": obj.class_name},
                )
            )
        sorted_candidates = sorted(now_candidates, key=lambda item: item.score, reverse=True)
        selected = sorted_candidates[: max(int(top_k), 0)]
        active_ids = {candidate.candidate_id for candidate in selected}
        for candidate in sorted_candidates:
            candidate.active = candidate.candidate_id in active_ids
        return sorted_candidates

    @staticmethod
    def _rule_context_bonus(
        *,
        obj: ObjectNode,
        place: PlaceNode | None,
        semantic_rules: list[SemanticRule],
    ) -> float:
        if not semantic_rules:
            return 0.0
        best = 0.0
        for rule in semantic_rules:
            hint = dict(rule.planner_hint)
            bonus = 0.0
            preferred_room = str(hint.get("preferred_room", "")).strip().lower()
            if preferred_room != "" and place is not None and place.room_id.lower() == preferred_room:
                bonus += 0.06
            preferred_place = str(hint.get("preferred_place_id", "")).strip().lower()
            if preferred_place != "" and place is not None and place.place_id.lower() == preferred_place:
                bonus += 0.04
            preferred_object = str(hint.get("preferred_object_id", "")).strip().lower()
            if preferred_object != "" and obj.object_id.lower() == preferred_object:
                bonus += 0.04
            support_classes = [str(item).strip().lower() for item in hint.get("preferred_support_classes", []) if str(item).strip() != ""]
            surface_class = str(obj.metadata.get("support_surface_class", "")).strip().lower()
            if support_classes and surface_class in support_classes:
                bonus += 0.04
            best = max(best, min(bonus, 0.10))
        return best
