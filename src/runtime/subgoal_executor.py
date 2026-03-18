from __future__ import annotations

from typing import Any

import numpy as np

from ipc.messages import ActionCommand
from locomotion.types import CommandEvaluation, ObstacleDefenseConfig, ObstacleDefenseResult
from locomotion.worker import LocomotionWorker
from runtime.planning_session import ExecutionObservation, TrajectoryUpdate
from schemas.commands import LocomotionProposal

SubgoalExecutionResult = LocomotionProposal


class SubgoalExecutor:
    """Compatibility wrapper around the proposal-only locomotion worker."""

    def __init__(
        self,
        args,
        *,
        planning_session: Any | None = None,
        follower: Any | None = None,
        worker: LocomotionWorker | None = None,
    ) -> None:
        del planning_session, follower
        self.args = args
        self._worker = worker or LocomotionWorker(args)

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        self._worker.initialize(simulation_app, stage)

    def shutdown(self) -> None:
        self._worker.shutdown()

    def command(self) -> np.ndarray:
        return self._worker.command()

    def empty_update(
        self,
        *,
        frame_idx: int,
        action_command: ActionCommand | None,
        error: str = "",
        stop: bool = False,
    ) -> TrajectoryUpdate:
        return self._worker.empty_update(
            frame_idx=frame_idx,
            action_command=action_command,
            error=error,
            stop=stop,
        )

    def step(
        self,
        *,
        frame_idx: int,
        observation: ExecutionObservation | None,
        action_command: ActionCommand | None,
        trajectory_update: TrajectoryUpdate,
        robot_pos_world: np.ndarray,
        robot_lin_vel_world: np.ndarray,
        robot_ang_vel_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> SubgoalExecutionResult:
        return self._worker.execute(
            frame_idx=frame_idx,
            observation=observation,
            action_command=action_command,
            trajectory_update=trajectory_update,
            robot_pos_world=robot_pos_world,
            robot_lin_vel_world=robot_lin_vel_world,
            robot_ang_vel_world=robot_ang_vel_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )
