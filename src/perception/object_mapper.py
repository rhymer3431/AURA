from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.models import ObsObject, pose3


@dataclass(frozen=True)
class Detection2D:
    class_name: str
    world_pose_xyz: tuple[float, float, float]
    timestamp: float
    confidence: float
    track_id: str = ""
    room_id: str = ""
    movable: bool = True
    state: str = "visible"
    embedding_id: str = ""
    snapshots: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ObjectMapper:
    def to_obs_object(self, detection: Detection2D) -> ObsObject:
        return ObsObject(
            class_name=detection.class_name,
            pose=pose3(detection.world_pose_xyz),
            timestamp=float(detection.timestamp),
            confidence=float(detection.confidence),
            track_id=detection.track_id,
            room_id=detection.room_id,
            movable=bool(detection.movable),
            state=detection.state,
            embedding_id=detection.embedding_id,
            snapshots=list(detection.snapshots),
            metadata=dict(detection.metadata),
        )
