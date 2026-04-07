from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnrichmentResult:
    observation: object | None
    perception_result: object | None
    memory_result: object | None
    enriched_batch: object | None


class EnrichmentService:
    def __init__(self, *, planner_coordinator, world_state_port) -> None:  # noqa: ANN001
        self._planner_coordinator = planner_coordinator
        self._world_state = world_state_port

    def enrich(
        self,
        *,
        frame_event,
        retrieve_memory: bool,
        instruction: str,
        task,
    ) -> EnrichmentResult:  # noqa: ANN001
        observation, perception_result, memory_result = self._planner_coordinator.enrich_observation(
            frame_event=frame_event,
            retrieve_memory=retrieve_memory,
            instruction=instruction,
        )
        enriched_batch = None if perception_result is None else perception_result.batch
        self._world_state.record_perception(
            enriched_batch,
            summary=None if perception_result is None else perception_result.summary,
        )
        self._world_state.record_memory_context(
            None if observation is None else observation.memory_context,
            summary=None if memory_result is None else memory_result.summary,
            task=task,
        )
        return EnrichmentResult(
            observation=observation,
            perception_result=perception_result,
            memory_result=memory_result,
            enriched_batch=enriched_batch,
        )
