from __future__ import annotations

from collections import deque
from typing import Iterable

from .models import ObsObject, TemporalEvent, pose3


class TemporalMemoryStore:
    def __init__(self, *, max_events: int = 256) -> None:
        self._events: deque[TemporalEvent] = deque(maxlen=max_events)

    def record_event(
        self,
        event_type: str,
        *,
        timestamp: float,
        track_id: str = "",
        person_id: str = "",
        object_id: str = "",
        pose: tuple[float, float, float] | list[float] | None = None,
        payload: dict[str, object] | None = None,
    ) -> TemporalEvent:
        event = TemporalEvent(
            event_type=str(event_type),
            timestamp=float(timestamp),
            track_id=str(track_id),
            person_id=str(person_id),
            object_id=str(object_id),
            pose=None if pose is None else pose3(pose),
            payload=dict(payload or {}),
        )
        self._events.append(event)
        return event

    def remember_observation(self, observation: ObsObject, *, object_id: str = "") -> TemporalEvent:
        return self.record_event(
            "object_observation",
            timestamp=float(observation.timestamp),
            track_id=observation.track_id,
            person_id=str(observation.metadata.get("person_id", "")),
            object_id=object_id,
            pose=observation.pose,
            payload={"class_name": observation.class_name, "confidence": float(observation.confidence)},
        )

    def recent_events(
        self,
        *,
        event_type: str = "",
        track_id: str = "",
        person_id: str = "",
        max_age_sec: float | None = None,
        now: float | None = None,
        limit: int | None = None,
    ) -> list[TemporalEvent]:
        cutoff = None
        if max_age_sec is not None:
            current = float(now if now is not None else self._events[-1].timestamp if self._events else 0.0)
            cutoff = current - float(max_age_sec)

        filtered: list[TemporalEvent] = []
        for event in reversed(self._events):
            if event_type != "" and event.event_type != event_type:
                continue
            if track_id != "" and event.track_id != track_id:
                continue
            if person_id != "" and event.person_id != person_id:
                continue
            if cutoff is not None and event.timestamp < cutoff:
                continue
            filtered.append(event)
            if limit is not None and len(filtered) >= int(limit):
                break
        return list(filtered)

    def last_event(self, *, event_type: str = "", track_id: str = "", person_id: str = "") -> TemporalEvent | None:
        events = self.recent_events(event_type=event_type, track_id=track_id, person_id=person_id, limit=1)
        return events[0] if events else None

    def reacquire_track(self, track_id: str, *, now: float, max_age_sec: float = 6.0) -> list[TemporalEvent]:
        return self.recent_events(track_id=track_id, max_age_sec=max_age_sec, now=now, limit=5)

    def reacquire_person(self, person_id: str, *, now: float, max_age_sec: float = 6.0) -> list[TemporalEvent]:
        return self.recent_events(person_id=person_id, max_age_sec=max_age_sec, now=now, limit=5)

    def __iter__(self) -> Iterable[TemporalEvent]:
        return iter(self._events)
