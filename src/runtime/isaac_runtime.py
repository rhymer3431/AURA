from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from common.geometry import quat_wxyz_to_yaw
from ipc.messages import ActionStatus

from .planning_session import PlanningSession, TrajectoryUpdate
from .supervisor import Supervisor, SupervisorConfig


@dataclass
class RuntimeStepResult:
    action_status: ActionStatus
    trajectory_update: TrajectoryUpdate


class IsaacRuntime:
    def __init__(self, args: argparse.Namespace, *, supervisor: Supervisor | None = None, planning_session: PlanningSession | None = None) -> None:
        self.args = args
        self.supervisor = supervisor or Supervisor()
        self.planning_session = planning_session or PlanningSession(args)
        self._active_command = None

    def initialize(self, simulation_app, stage) -> None:
        self.planning_session.initialize(simulation_app, stage)

    def submit_task(self, command_text: str, *, target_json: dict[str, object] | None = None, speaker_id: str = "") -> None:
        self.supervisor.submit_task(command_text, target_json=target_json, speaker_id=speaker_id)

    def update(self, frame_id: int, *, robot_pos_world: np.ndarray, robot_quat_wxyz: np.ndarray, env=None) -> RuntimeStepResult:  # noqa: ANN001
        robot_pose = tuple(float(v) for v in np.asarray(robot_pos_world, dtype=np.float32).reshape(-1)[:3])
        command = self.supervisor.step(now=time.time(), robot_pose=robot_pose)
        self._active_command = command
        trajectory_update = self.planning_session.update(
            frame_id,
            action_command=command,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(quat_wxyz_to_yaw(np.asarray(robot_quat_wxyz, dtype=np.float32))),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
            env=env,
        )
        status = ActionStatus(
            command_id=command.command_id if command is not None else "",
            state="running" if command is not None else "idle",
            success=False,
            robot_pose_xyz=robot_pose,
        )
        return RuntimeStepResult(action_status=status, trajectory_update=trajectory_update)

    def shutdown(self) -> None:
        self.planning_session.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Direct Isaac runtime scaffold.")
    parser.add_argument("--command", type=str, default="")
    parser.add_argument("--memory-db-path", type=str, default="state/memory/memory.sqlite")
    args = parser.parse_args(argv)
    runtime = IsaacRuntime(args, supervisor=Supervisor(config=SupervisorConfig(memory_db_path=args.memory_db_path)))
    if str(args.command).strip() != "":
        runtime.submit_task(str(args.command))
    print(f"[ISAAC_RUNTIME] configured memory_db_path={args.memory_db_path}")
    return 0
