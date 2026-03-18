from __future__ import annotations

from dataclasses import dataclass

from schemas.events import FrameEvent
from schemas.world_state import TaskSnapshot, WorldStateSnapshot


@dataclass(frozen=True)
class DecisionDirective:
    process_perception: bool
    retrieve_memory: bool
    use_manual_command: bool
    route_task_command: bool
    reason: str = ""


class DecisionEngine:
    def evaluate(
        self,
        *,
        world_state: WorldStateSnapshot,
        task: TaskSnapshot,
        frame_event: FrameEvent,
        manual_command_present: bool,
        active_memory_instruction: str,
    ) -> DecisionDirective:
        process_perception = bool(frame_event.batch is not None and frame_event.observation is not None)
        retrieve_memory = process_perception and str(active_memory_instruction).strip() != ""
        route_task_command = not bool(manual_command_present)
        reason = "manual_command"
        if route_task_command:
            reason = "task_orchestrator"
        if not process_perception:
            reason = "sensor_unavailable"
        if retrieve_memory:
            reason = "memory_aware_planning"
        return DecisionDirective(
            process_perception=process_perception,
            retrieve_memory=retrieve_memory,
            use_manual_command=bool(manual_command_present),
            route_task_command=route_task_command,
            reason=reason,
        )
