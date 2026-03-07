from __future__ import annotations

from memory.models import ObsObject


class ObservationFuser:
    def fuse(self, observations: list[ObsObject]) -> list[ObsObject]:
        deduped: dict[tuple[str, str], ObsObject] = {}
        for observation in observations:
            key = (observation.track_id, observation.class_name)
            existing = deduped.get(key)
            if existing is None or observation.confidence >= existing.confidence:
                deduped[key] = observation
        return list(deduped.values())
