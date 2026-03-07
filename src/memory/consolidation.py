from __future__ import annotations

from .episodic_store import EpisodicMemoryStore
from .semantic_store import SemanticMemoryStore
from services.semantic_consolidation import SemanticConsolidationService


class MemoryConsolidator:
    def __init__(self, episodic_store: EpisodicMemoryStore, semantic_store: SemanticMemoryStore) -> None:
        self._episodic = episodic_store
        self._semantic = semantic_store
        self._service = SemanticConsolidationService(semantic_store)

    def consolidate_episode(self, episode_id: str) -> None:
        episode = self._episodic.get(episode_id)
        if episode is None:
            return
        self._service.consolidate(episode)
