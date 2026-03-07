from __future__ import annotations

import math

from .models import ObjectNode, ObsObject, pose3


def xy_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return math.hypot(dx, dy)


def embedding_similarity(left: str, right: str) -> float:
    if left == "" or right == "":
        return 0.0
    return 1.0 if left == right else 0.0


def association_score(observation: ObsObject, candidate: ObjectNode, *, now: float) -> tuple[float, dict[str, float]]:
    obs_pose = pose3(observation.pose)
    cand_pose = pose3(candidate.last_pose)
    pose_delta = xy_distance(obs_pose, cand_pose)
    time_gap = max(float(now) - float(candidate.last_seen), 0.0)
    track_bonus = 1.0 if observation.track_id != "" and observation.track_id == candidate.track_id else 0.0
    class_bonus = 1.0 if observation.class_name == candidate.class_name else 0.0
    embedding_bonus = embedding_similarity(observation.embedding_id, candidate.embedding_id)
    distance_term = max(0.0, 1.0 - min(pose_delta / 2.0, 1.0))
    recency_term = math.exp(-time_gap / 60.0)
    score = (
        0.40 * track_bonus
        + 0.25 * class_bonus
        + 0.15 * distance_term
        + 0.10 * recency_term
        + 0.10 * embedding_bonus
    )
    return score, {
        "track_bonus": track_bonus,
        "class_bonus": class_bonus,
        "distance_term": distance_term,
        "recency_term": recency_term,
        "embedding_bonus": embedding_bonus,
    }
