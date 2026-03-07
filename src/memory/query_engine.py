from __future__ import annotations

from .models import RecallQuery, RecallResult
from .semantic_store import SemanticMemoryStore
from .spatial_store import SpatialMemoryStore
from .working_memory import WorkingMemory


class MemoryQueryEngine:
    def __init__(
        self,
        spatial_store: SpatialMemoryStore,
        semantic_store: SemanticMemoryStore,
        working_memory: WorkingMemory,
    ) -> None:
        self._spatial = spatial_store
        self._semantic = semantic_store
        self._working = working_memory

    def recall_object(
        self,
        query: RecallQuery,
        *,
        current_pose: tuple[float, float, float] | None = None,
    ) -> RecallResult:
        objects = self._spatial.recall_objects(class_name=query.target_class, query_text=query.query_text)
        rules = self._semantic.matching_rules(
            intent=query.intent,
            target_class=query.target_class,
            room_id=query.room_id,
        )
        candidates = self._working.select_candidates(
            query_text=query.query_text,
            objects=objects,
            places=self._spatial.places,
            semantic_rules=rules,
            current_pose=current_pose,
            target_class=query.target_class,
            room_id=query.room_id,
            top_k=query.top_k,
        )
        selected = next((candidate for candidate in candidates if candidate.active), None)
        selected_object = self._spatial.objects.get(selected.object_id) if selected is not None else None
        selected_place = self._spatial.place_for_object(selected.object_id) if selected is not None else None
        return RecallResult(
            query=query,
            candidates=candidates,
            selected_object=selected_object,
            selected_place=selected_place,
        )
