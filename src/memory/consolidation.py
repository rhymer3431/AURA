from __future__ import annotations

from .episodic_store import EpisodicMemoryStore
from .semantic_store import SemanticMemoryStore


class MemoryConsolidator:
    def __init__(self, episodic_store: EpisodicMemoryStore, semantic_store: SemanticMemoryStore) -> None:
        self._episodic = episodic_store
        self._semantic = semantic_store

    def consolidate_episode(self, episode_id: str) -> None:
        episode = self._episodic.get(episode_id)
        if episode is None:
            return
        target_class = str(episode.target_json.get("target_class", "")).strip().lower()
        room_id = str(episode.target_json.get("room_id", "")).strip().lower()
        if target_class == "":
            return
        rule_key = ":".join(token for token in ("find", target_class, room_id) if token != "")
        description = f"find:{target_class}:{room_id or 'any'}"
        self._semantic.remember_rule(rule_key, description, succeeded=bool(episode.success))
