"""World-model and memory path wrappers for the canonical runtime flow."""

from __future__ import annotations

from dataclasses import dataclass, field

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from memory.models import MemoryContextBundle
from runtime.supervisor import Supervisor
from services.memory_service import MemoryService


@dataclass
class MemoryWritePath:
    """Explicit write-side facade over the mixed memory service API."""

    memory_service: MemoryService

    def record_perception_frame(self, **kwargs):  # noqa: ANN003
        return self.memory_service.record_perception_frame(**kwargs)

    def set_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self.memory_service.set_planner_task(**kwargs)

    def clear_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self.memory_service.clear_planner_task(**kwargs)


@dataclass
class MemoryReadPath:
    """Explicit read-side facade over the mixed memory service API."""

    memory_service: MemoryService

    def build_memory_context(
        self,
        *,
        instruction: str,
        current_pose: tuple[float, float, float] | None = None,
        max_text_lines: int = 5,
        max_keyframes: int = 2,
    ) -> MemoryContextBundle | None:
        return self.memory_service.build_memory_context(
            instruction=instruction,
            current_pose=current_pose,
            max_text_lines=max_text_lines,
            max_keyframes=max_keyframes,
        )

    def recall_object(self, **kwargs):  # noqa: ANN003
        return self.memory_service.recall_object(**kwargs)

    def preview_object_recall(self, **kwargs):  # noqa: ANN003
        return self.memory_service.preview_object_recall(**kwargs)


@dataclass
class WorldModelModule:
    """Perception ingress and memory access facade for navigation runtime."""

    supervisor: Supervisor
    memory_write: MemoryWritePath = field(init=False)
    memory_read: MemoryReadPath = field(init=False)

    def __post_init__(self) -> None:
        self.memory_write = MemoryWritePath(self.supervisor.memory_service)
        self.memory_read = MemoryReadPath(self.supervisor.memory_service)

    def update(self, batch: IsaacObservationBatch) -> IsaacObservationBatch:
        return self.supervisor.update_world_model(batch)

    def snapshot(self) -> dict[str, object]:
        return self.supervisor.snapshot()
