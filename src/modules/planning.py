"""Planning module wrappers for planner-owned execution updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipc.messages import ActionCommand
from planning.coordinator import PlanningCoordinator
from runtime.planning_session import ExecutionObservation, PlanningSession, TrajectoryUpdate


@dataclass
class PlanningModule:
    """Planner facade that keeps backend ownership inside PlanningSession."""

    session: PlanningSession

    def plan(
        self,
        observation: ExecutionObservation,
        *,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> TrajectoryUpdate:
        return self.session.plan(
            observation,
            action_command=action_command,
            robot_pos_world=robot_pos_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )

    @property
    def coordinator_type(self) -> type[PlanningCoordinator]:
        return PlanningCoordinator
