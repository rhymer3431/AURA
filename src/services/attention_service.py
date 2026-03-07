from __future__ import annotations

from memory.models import ObsObject
from perception.speaker_events import SpeakerEvent


class AttentionService:
    def __init__(self) -> None:
        self._latest_event: SpeakerEvent | None = None
        self._bound_track_id = ""

    @property
    def bound_track_id(self) -> str:
        return self._bound_track_id

    def record_event(self, event: SpeakerEvent) -> None:
        self._latest_event = event

    def current_event(self) -> SpeakerEvent | None:
        return self._latest_event

    def bind_person(self, observations: list[ObsObject]) -> str:
        for observation in observations:
            if observation.class_name.lower() != "person":
                continue
            if observation.track_id != "":
                self._bound_track_id = observation.track_id
                return self._bound_track_id
        return self._bound_track_id
