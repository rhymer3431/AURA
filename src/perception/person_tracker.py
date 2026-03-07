from __future__ import annotations

from dataclasses import dataclass

from memory.models import ObsObject, pose3


@dataclass
class PersonTrack:
    track_id: str
    last_pose: tuple[float, float, float]
    last_seen: float
    confidence: float = 1.0
    reid_id: str = ""


class PersonTracker:
    def __init__(self) -> None:
        self._tracks: dict[str, PersonTrack] = {}

    def update(self, observations: list[ObsObject]) -> None:
        for observation in observations:
            if observation.class_name.lower() != "person" or observation.track_id == "":
                continue
            self._tracks[observation.track_id] = PersonTrack(
                track_id=observation.track_id,
                last_pose=pose3(observation.pose),
                last_seen=float(observation.timestamp),
                confidence=float(observation.confidence),
                reid_id=observation.embedding_id,
            )

    def get(self, track_id: str) -> PersonTrack | None:
        return self._tracks.get(track_id)

    def all_tracks(self) -> list[PersonTrack]:
        return sorted(self._tracks.values(), key=lambda item: item.last_seen, reverse=True)
