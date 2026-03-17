"""Mission module wrappers for orchestration and high-level actions."""

from __future__ import annotations

from dataclasses import dataclass

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from ipc.messages import ActionCommand, ActionStatus, TaskRequest
from mission.mission_manager import MissionManager
from runtime.supervisor import Supervisor


@dataclass
class MissionModule:
    """Mission facade over task orchestration and ingress wiring."""

    supervisor: Supervisor

    @property
    def manager(self) -> MissionManager:
        return self.supervisor.orchestrator  # type: ignore[return-value]

    def submit_task(
        self,
        command_text: str,
        *,
        target_json: dict[str, object] | None = None,
        speaker_id: str = "",
    ) -> TaskRequest:
        return self.supervisor.submit_task(command_text, target_json=target_json, speaker_id=speaker_id)

    def submit_task_request(self, request: TaskRequest) -> TaskRequest:
        return self.supervisor.submit_task_request(request)

    def update(self, batch: IsaacObservationBatch, *, publish: bool = True) -> IsaacObservationBatch:
        return self.supervisor.update_mission(batch, publish=publish)

    def propose_action(
        self,
        *,
        now: float,
        robot_pose: tuple[float, float, float],
        robot_yaw_rad: float | None = None,
        action_status: ActionStatus | None = None,
        publish: bool = False,
    ) -> ActionCommand | None:
        return self.supervisor.step(
            now=now,
            robot_pose=robot_pose,
            robot_yaw_rad=robot_yaw_rad,
            action_status=action_status,
            publish=publish,
        )

    def snapshot(self) -> dict[str, object]:
        return self.manager.snapshot()
