from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from pathlib import Path

from memory import (
    EpisodeRecord,
    EpisodicMemoryStore,
    MemoryConsolidator,
    MemoryQueryEngine,
    RecallQuery,
    SQLiteMemoryPersistence,
    SemanticMemoryStore,
    SpatialMemoryStore,
    TemporalMemoryStore,
    WorkingMemory,
)
from perception.speaker_events import SpeakerEvent


class MemoryService:
    def __init__(self, *, db_path: str | None = None) -> None:
        self.spatial_store = SpatialMemoryStore()
        self.temporal_store = TemporalMemoryStore()
        self.episodic_store = EpisodicMemoryStore()
        self.semantic_store = SemanticMemoryStore()
        self.working_memory = WorkingMemory()
        self.query_engine = MemoryQueryEngine(self.spatial_store, self.semantic_store, self.working_memory)
        self.consolidator = MemoryConsolidator(self.episodic_store, self.semantic_store)
        self.persistence = SQLiteMemoryPersistence(Path(db_path)) if db_path is not None else None
        if self.persistence is not None:
            self.persistence.initialize()
        self._active_episode_id = ""

    def observe_objects(self, observations) -> list[object]:  # noqa: ANN001
        results = []
        for observation in observations:
            result = self.spatial_store.associate_observation(observation)
            self.temporal_store.remember_observation(observation, object_id=result.object_node.object_id)
            results.append(result)
        return results

    def record_speaker_event(self, event: SpeakerEvent) -> None:
        self.temporal_store.record_event(
            "speaker_event",
            timestamp=float(event.timestamp),
            track_id=event.speaker_id,
            payload={"yaw_rad": float(event.direction_yaw_rad), "confidence": float(event.confidence), **event.metadata},
        )

    def start_episode(self, *, command_text: str, intent: str, target_json: dict[str, object]) -> str:
        episode_id = f"episode_{uuid.uuid4().hex[:12]}"
        self._active_episode_id = episode_id
        record = EpisodeRecord(
            episode_id=episode_id,
            command_text=str(command_text),
            intent=str(intent),
            target_json=dict(target_json),
            started_at=time.time(),
        )
        self.episodic_store.put(record)
        return episode_id

    def finish_episode(
        self,
        *,
        success: bool,
        failure_reason: str = "",
        recovery_actions: list[str] | None = None,
        summary_text: str = "",
    ) -> None:
        if self._active_episode_id == "":
            return
        record = self.episodic_store.get(self._active_episode_id)
        if record is None:
            return
        record.success = bool(success)
        record.failure_reason = str(failure_reason)
        record.recovery_actions = list(recovery_actions or [])
        record.summary_text = str(summary_text)
        record.ended_at = time.time()
        self.consolidator.consolidate_episode(record.episode_id)
        self._active_episode_id = ""

    def recall_object(
        self,
        *,
        query_text: str,
        target_class: str,
        intent: str,
        room_id: str = "",
        current_pose: tuple[float, float, float] | None = None,
    ):
        return self.query_engine.recall_object(
            RecallQuery(
                query_text=str(query_text),
                target_class=str(target_class),
                intent=str(intent),
                room_id=str(room_id),
            ),
            current_pose=current_pose,
        )

    def reacquire_follow_target(self, track_id: str, *, now: float, max_age_sec: float = 6.0):
        return self.temporal_store.reacquire_track(track_id, now=now, max_age_sec=max_age_sec)

    def persist_snapshot(self) -> int | None:
        if self.persistence is None:
            return None
        payload = {
            "places": [asdict(place) for place in self.spatial_store.places.values()],
            "objects": [asdict(obj) for obj in self.spatial_store.objects.values()],
            "semantic_rules": [asdict(rule) for rule in self.semantic_store.list()],
        }
        return self.persistence.save_snapshot("memory_service", payload)
