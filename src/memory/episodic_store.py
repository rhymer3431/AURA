from __future__ import annotations

from .models import EpisodeRecord


class EpisodicMemoryStore:
    def __init__(self) -> None:
        self._episodes: dict[str, EpisodeRecord] = {}

    def put(self, record: EpisodeRecord) -> None:
        self._episodes[record.episode_id] = record

    def get(self, episode_id: str) -> EpisodeRecord | None:
        return self._episodes.get(episode_id)

    def list(self) -> list[EpisodeRecord]:
        return sorted(self._episodes.values(), key=lambda item: item.started_at, reverse=True)
