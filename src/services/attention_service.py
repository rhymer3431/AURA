from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from common.geometry import wrap_to_pi
from memory.temporal_store import TemporalMemoryStore
from perception.person_tracker import PersonTrack
from perception.speaker_events import SpeakerEvent


@dataclass
class PendingSpeakerEvent:
    event: SpeakerEvent
    expires_at: float
    bound_person_id: str = ""
    bound_track_id: str = ""


class AttentionService:
    def __init__(self, temporal_store: TemporalMemoryStore, *, ttl_sec: float = 2.5) -> None:
        self._temporal = temporal_store
        self._ttl_sec = float(ttl_sec)
        self._pending_events: deque[PendingSpeakerEvent] = deque()
        self._bound_track_id = ""
        self._bound_person_id = ""

    @property
    def bound_track_id(self) -> str:
        return self._bound_track_id

    @property
    def bound_person_id(self) -> str:
        return self._bound_person_id

    def record_event(self, event: SpeakerEvent) -> None:
        timestamp = float(event.timestamp)
        self._pending_events.append(
            PendingSpeakerEvent(
                event=event,
                expires_at=timestamp + self._ttl_sec,
            )
        )

    def current_event(self, *, now: float | None = None) -> SpeakerEvent | None:
        self._prune(now=now)
        if not self._pending_events:
            return None
        return self._pending_events[-1].event

    def bind_person(self, person_tracks: list[PersonTrack], *, now: float) -> str:
        self._prune(now=now)
        if not self._pending_events:
            return self._bound_person_id
        pending = self._pending_events[-1]
        best_track = None
        best_score = -1.0
        for track in person_tracks:
            score = self._binding_score(track, pending.event, now=now)
            if score > best_score:
                best_track = track
                best_score = score
        if best_track is None or best_score < 0.35:
            return self._bound_person_id
        pending.bound_person_id = best_track.person_id
        pending.bound_track_id = best_track.track_id
        self._bound_person_id = best_track.person_id
        self._bound_track_id = best_track.track_id
        self._temporal.record_event(
            "speaker_binding",
            timestamp=float(now),
            track_id=best_track.track_id,
            person_id=best_track.person_id,
            pose=best_track.last_pose,
            payload={
                "score": float(best_score),
                "speaker_id": pending.event.speaker_id,
                "direction_yaw_rad": float(pending.event.direction_yaw_rad),
            },
        )
        return self._bound_person_id

    def _prune(self, *, now: float | None = None) -> None:
        current = float(now if now is not None else self._pending_events[-1].event.timestamp if self._pending_events else 0.0)
        while self._pending_events and self._pending_events[0].expires_at < current:
            self._pending_events.popleft()

    @staticmethod
    def _binding_score(track: PersonTrack, event: SpeakerEvent, *, now: float) -> float:
        recency = math.exp(-max(0.0, float(now) - float(track.last_seen)) / 2.0)
        yaw_fit = 0.5
        if track.last_yaw_hint is not None:
            delta = abs(float(wrap_to_pi(float(track.last_yaw_hint) - float(event.direction_yaw_rad))))
            yaw_fit = max(0.0, 1.0 - delta / math.pi)
        confidence = min(max(float(track.confidence), 0.0), 1.0)
        return 0.45 * yaw_fit + 0.35 * recency + 0.20 * confidence
