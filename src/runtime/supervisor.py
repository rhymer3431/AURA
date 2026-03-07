from __future__ import annotations

from dataclasses import dataclass

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter, IsaacBridgeAdapterConfig, IsaacObservationBatch
from ipc.base import MessageBus
from ipc.inproc_bus import InprocBus
from ipc.messages import ActionCommand, ActionStatus, TaskRequest
from services.memory_service import MemoryService
from services.task_orchestrator import TaskOrchestrator


@dataclass(frozen=True)
class SupervisorConfig:
    memory_db_path: str = "state/memory/memory.sqlite"


class Supervisor:
    def __init__(
        self,
        *,
        bus: MessageBus | None = None,
        config: SupervisorConfig | None = None,
        memory_service: MemoryService | None = None,
        orchestrator: TaskOrchestrator | None = None,
    ) -> None:
        self.bus = bus or InprocBus()
        self.config = config or SupervisorConfig()
        self.memory_service = memory_service or MemoryService(db_path=self.config.memory_db_path)
        self.orchestrator = orchestrator or TaskOrchestrator(self.memory_service)
        self.bridge = IsaacBridgeAdapter(self.bus, IsaacBridgeAdapterConfig())

    def submit_task(self, command_text: str, *, target_json: dict[str, object] | None = None, speaker_id: str = "") -> TaskRequest:
        request = TaskRequest(command_text=command_text, target_json=dict(target_json or {}), speaker_id=speaker_id)
        self.bridge.publish_task_request(request)
        self.orchestrator.submit_task(request)
        return request

    def ingest(self, batch: IsaacObservationBatch) -> None:
        self.bridge.publish_observation_batch(batch)
        if batch.speaker_events:
            for event in batch.speaker_events:
                self.orchestrator.on_speaker_event(event)
        if batch.observations:
            self.orchestrator.on_observations(batch.observations)

    def step(
        self,
        *,
        now: float,
        robot_pose: tuple[float, float, float],
        action_status: ActionStatus | None = None,
    ) -> ActionCommand | None:
        if action_status is not None:
            self.bridge.publish_status(action_status)
        command = self.orchestrator.step(now=now, robot_pose=robot_pose, action_status=action_status)
        if command is not None:
            self.bus.publish(self.bridge.config.command_topic, command)
        return command

    def snapshot(self) -> dict[str, object]:
        return self.orchestrator.snapshot()
