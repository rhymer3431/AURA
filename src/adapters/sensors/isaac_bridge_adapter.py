from __future__ import annotations

from dataclasses import dataclass, field

from ipc.base import MessageBus
from ipc.messages import ActionCommand, ActionStatus, FrameHeader, TaskRequest
from memory.models import ObsObject
from perception.speaker_events import SpeakerEvent


@dataclass(frozen=True)
class IsaacBridgeAdapterConfig:
    observation_topic: str = "isaac.observation"
    command_topic: str = "isaac.command"
    status_topic: str = "isaac.status"
    task_topic: str = "isaac.task"


@dataclass
class IsaacObservationBatch:
    frame_header: FrameHeader
    robot_pose_xyz: tuple[float, float, float]
    observations: list[ObsObject] = field(default_factory=list)
    speaker_events: list[SpeakerEvent] = field(default_factory=list)


class IsaacBridgeAdapter:
    def __init__(self, bus: MessageBus, config: IsaacBridgeAdapterConfig | None = None) -> None:
        self._bus = bus
        self.config = config or IsaacBridgeAdapterConfig()
        self._latest_batch: IsaacObservationBatch | None = None

    @property
    def latest_batch(self) -> IsaacObservationBatch | None:
        return self._latest_batch

    def publish_observation_batch(self, batch: IsaacObservationBatch) -> None:
        self._latest_batch = batch
        self._bus.publish(self.config.observation_topic, batch.frame_header)

    def publish_task_request(self, request: TaskRequest) -> None:
        self._bus.publish(self.config.task_topic, request)

    def publish_status(self, status: ActionStatus) -> None:
        self._bus.publish(self.config.status_topic, status)

    def drain_commands(self) -> list[ActionCommand]:
        return [record.message for record in self._bus.poll(self.config.command_topic, max_items=32)]
