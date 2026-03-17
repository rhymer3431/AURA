"""Mission-level facade over the legacy task orchestrator."""

from __future__ import annotations

from ipc.messages import ActionCommand, ActionStatus

from services.task_orchestrator import TaskOrchestrator, TaskOrchestratorConfig

MissionManagerConfig = TaskOrchestratorConfig


class MissionManager(TaskOrchestrator):
    """Compatibility-first alias for the mission module boundary."""

    def consume_observations(self, observations) -> None:  # noqa: ANN001
        self.on_observations(observations)

    def update(
        self,
        *,
        now: float,
        robot_pose: tuple[float, float, float] | None = None,
        robot_yaw_rad: float | None = None,
        action_status: ActionStatus | None = None,
    ) -> ActionCommand | None:
        return self.step(
            now=now,
            robot_pose=robot_pose,
            robot_yaw_rad=robot_yaw_rad,
            action_status=action_status,
        )
