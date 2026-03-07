from __future__ import annotations

from dataclasses import dataclass, field

from memory.models import ObsObject, pose3

from .reid_store import ReIdStore


@dataclass
class PersonTrack:
    person_id: str
    track_id: str
    last_track_id: str
    last_pose: tuple[float, float, float]
    last_seen: float
    confidence: float = 1.0
    reid_id: str = ""
    appearance_signature: str = ""
    last_yaw_hint: float | None = None
    score_breakdown: dict[str, float] = field(default_factory=dict)


class PersonTracker:
    def __init__(self, *, reid_store: ReIdStore | None = None) -> None:
        self._tracks_by_person: dict[str, PersonTrack] = {}
        self._person_by_track: dict[str, str] = {}
        self._reid = reid_store or ReIdStore()

    def update(self, observations: list[ObsObject], *, speaker_yaw_hint: float | None = None) -> list[PersonTrack]:
        updated: list[PersonTrack] = []
        for observation in observations:
            if observation.class_name.lower() != "person" or observation.track_id == "":
                continue
            appearance_signature = str(
                observation.metadata.get("appearance_signature")
                or observation.metadata.get("color_hint")
                or observation.embedding_id
            ).strip()
            yaw_hint = observation.metadata.get("bearing_yaw_rad")
            person_identity, match = self._reid.assign(
                track_id=observation.track_id,
                pose=pose3(observation.pose),
                timestamp=float(observation.timestamp),
                confidence=float(observation.confidence),
                embedding_id=str(observation.embedding_id),
                appearance_signature=appearance_signature,
                yaw_hint=None if yaw_hint is None else float(yaw_hint),
                speaker_yaw_hint=speaker_yaw_hint,
            )
            observation.metadata["person_id"] = person_identity.person_id
            observation.metadata["appearance_signature"] = appearance_signature
            track = PersonTrack(
                person_id=person_identity.person_id,
                track_id=observation.track_id,
                last_track_id=observation.track_id,
                last_pose=pose3(observation.pose),
                last_seen=float(observation.timestamp),
                confidence=float(observation.confidence),
                reid_id=observation.embedding_id,
                appearance_signature=appearance_signature,
                last_yaw_hint=None if yaw_hint is None else float(yaw_hint),
                score_breakdown={
                    "embedding_match": float(match.embedding_match),
                    "spatial_continuity": float(match.spatial_continuity),
                    "recency": float(match.recency),
                    "speaker_yaw_fit": float(match.speaker_yaw_fit),
                    "score": float(match.score),
                },
            )
            self._tracks_by_person[track.person_id] = track
            self._person_by_track[track.track_id] = track.person_id
            updated.append(track)
        return sorted(updated, key=lambda item: item.last_seen, reverse=True)

    def get(self, track_or_person_id: str) -> PersonTrack | None:
        direct = self._tracks_by_person.get(str(track_or_person_id))
        if direct is not None:
            return direct
        person_id = self._person_by_track.get(str(track_or_person_id), "")
        if person_id == "":
            return None
        return self._tracks_by_person.get(person_id)

    def get_by_person_id(self, person_id: str) -> PersonTrack | None:
        return self._tracks_by_person.get(str(person_id))

    def person_id_for_track(self, track_id: str) -> str:
        return self._person_by_track.get(str(track_id), "")

    def all_tracks(self) -> list[PersonTrack]:
        return sorted(self._tracks_by_person.values(), key=lambda item: item.last_seen, reverse=True)
