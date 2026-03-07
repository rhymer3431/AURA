from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReIdMatch:
    source_track_id: str
    target_track_id: str
    similarity: float


class ReIdResolver:
    def match(self, source_embedding_id: str, candidate_embedding_id: str) -> float:
        if source_embedding_id == "" or candidate_embedding_id == "":
            return 0.0
        return 1.0 if source_embedding_id == candidate_embedding_id else 0.0
