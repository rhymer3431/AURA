from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

from common.geometry import wrap_to_pi


@dataclass
class ReIdIdentity:
    person_id: str
    last_track_id: str
    last_pose: tuple[float, float, float]
    last_seen: float
    confidence: float
    embedding_id: str = ""
    appearance_signature: str = ""
    last_yaw_hint: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReIdMatch:
    person_id: str
    score: float
    embedding_match: float
    spatial_continuity: float
    recency: float
    speaker_yaw_fit: float


class ReIdStore:
    def __init__(self, *, match_threshold: float = 0.45) -> None:
        self._match_threshold = float(match_threshold)
        self._identities: dict[str, ReIdIdentity] = {}

    def assign(
        self,
        *,
        track_id: str,
        pose: tuple[float, float, float],
        timestamp: float,
        confidence: float,
        embedding_id: str = "",
        appearance_signature: str = "",
        yaw_hint: float | None = None,
        speaker_yaw_hint: float | None = None,
    ) -> tuple[ReIdIdentity, ReIdMatch]:
        candidates = self.candidate_matches(
            pose=pose,
            timestamp=timestamp,
            embedding_id=embedding_id,
            appearance_signature=appearance_signature,
            speaker_yaw_hint=speaker_yaw_hint,
        )
        if candidates and candidates[0].score >= self._match_threshold:
            identity = self._identities[candidates[0].person_id]
            identity.last_track_id = str(track_id)
            identity.last_pose = tuple(float(v) for v in pose[:3])
            identity.last_seen = float(timestamp)
            identity.confidence = float(confidence)
            if embedding_id != "":
                identity.embedding_id = str(embedding_id)
            if appearance_signature != "":
                identity.appearance_signature = str(appearance_signature)
            if yaw_hint is not None:
                identity.last_yaw_hint = float(yaw_hint)
            return identity, candidates[0]

        identity = ReIdIdentity(
            person_id=f"person_{uuid.uuid4().hex[:10]}",
            last_track_id=str(track_id),
            last_pose=tuple(float(v) for v in pose[:3]),
            last_seen=float(timestamp),
            confidence=float(confidence),
            embedding_id=str(embedding_id),
            appearance_signature=str(appearance_signature),
            last_yaw_hint=None if yaw_hint is None else float(yaw_hint),
        )
        self._identities[identity.person_id] = identity
        return identity, ReIdMatch(
            person_id=identity.person_id,
            score=1.0,
            embedding_match=1.0 if embedding_id != "" else 0.0,
            spatial_continuity=1.0,
            recency=1.0,
            speaker_yaw_fit=1.0 if yaw_hint is not None and speaker_yaw_hint is not None else 0.0,
        )

    def candidate_matches(
        self,
        *,
        pose: tuple[float, float, float],
        timestamp: float,
        embedding_id: str = "",
        appearance_signature: str = "",
        speaker_yaw_hint: float | None = None,
    ) -> list[ReIdMatch]:
        matches: list[ReIdMatch] = []
        for identity in self._identities.values():
            spatial_continuity = self._spatial_continuity(identity.last_pose, pose)
            recency = math.exp(-max(0.0, float(timestamp) - float(identity.last_seen)) / 3.0)
            embedding_match = self._embedding_match(identity, embedding_id=embedding_id, appearance_signature=appearance_signature)
            yaw_fit = self._yaw_fit(identity.last_yaw_hint, speaker_yaw_hint)
            if embedding_id != "" or appearance_signature != "":
                score = 0.45 * embedding_match + 0.30 * spatial_continuity + 0.20 * recency + 0.05 * yaw_fit
            else:
                score = 0.65 * spatial_continuity + 0.35 * recency
            matches.append(
                ReIdMatch(
                    person_id=identity.person_id,
                    score=float(score),
                    embedding_match=float(embedding_match),
                    spatial_continuity=float(spatial_continuity),
                    recency=float(recency),
                    speaker_yaw_fit=float(yaw_fit),
                )
            )
        return sorted(matches, key=lambda item: item.score, reverse=True)

    def get(self, person_id: str) -> ReIdIdentity | None:
        return self._identities.get(str(person_id))

    def all_identities(self) -> list[ReIdIdentity]:
        return sorted(self._identities.values(), key=lambda item: item.last_seen, reverse=True)

    @staticmethod
    def _spatial_continuity(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        distance = (dx * dx + dy * dy) ** 0.5
        return 1.0 / (1.0 + distance)

    @staticmethod
    def _embedding_match(identity: ReIdIdentity, *, embedding_id: str, appearance_signature: str) -> float:
        if embedding_id != "" and identity.embedding_id != "":
            return 1.0 if identity.embedding_id == embedding_id else 0.0
        if appearance_signature != "" and identity.appearance_signature != "":
            return 0.9 if identity.appearance_signature == appearance_signature else 0.0
        return 0.0

    @staticmethod
    def _yaw_fit(candidate_yaw: float | None, speaker_yaw_hint: float | None) -> float:
        if candidate_yaw is None or speaker_yaw_hint is None:
            return 0.0
        delta = abs(float(wrap_to_pi(float(candidate_yaw) - float(speaker_yaw_hint))))
        return max(0.0, 1.0 - delta / math.pi)
