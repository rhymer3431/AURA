from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from common.geometry import within_xy_radius, wrap_to_pi, xy_distance
from control.trajectory_tracker import TrajectoryTracker, TrajectoryTrackerConfig
from ipc.messages import ActionCommand, ActionStatus

from .planning_session import PlannerStats, PlanningSession, TrajectoryUpdate


@dataclass(frozen=True)
class CommandEvaluation:
    force_stop: bool
    goal_distance_m: float
    yaw_error_rad: float
    reached_goal: bool


@dataclass(frozen=True)
class SubgoalExecutionResult:
    command_vector: np.ndarray
    trajectory_update: TrajectoryUpdate
    evaluation: CommandEvaluation
    status: ActionStatus | None


class SubgoalExecutor:
    def __init__(
        self,
        args,
        *,
        planning_session: PlanningSession | None = None,
    ) -> None:
        self.args = args
        self._planning_session = planning_session or PlanningSession(args)
        self._tracker = TrajectoryTracker(
            TrajectoryTrackerConfig(
                max_vx=float(args.cmd_max_vx),
                max_vy=float(args.cmd_max_vy),
                max_wz=float(args.cmd_max_wz),
                lookahead_distance_m=float(args.lookahead_distance_m),
                heading_slowdown_rad=float(args.heading_slowdown_rad),
                traj_stale_timeout_sec=float(args.traj_stale_timeout_sec),
                cmd_accel_limit=float(args.cmd_accel_limit),
                cmd_yaw_accel_limit=float(args.cmd_yaw_accel_limit),
            )
        )
        self._last_applied_plan_version = -1
        self._planner_yaw_target_rad: float | None = None
        self._command = np.zeros(3, dtype=np.float32)

    @property
    def planning_session(self) -> PlanningSession:
        return self._planning_session

    def initialize(self, simulation_app, stage) -> None:
        self._planning_session.initialize(simulation_app, stage)

    def shutdown(self) -> None:
        self._planning_session.shutdown()

    def command(self) -> np.ndarray:
        return self._command.copy()

    def step(
        self,
        *,
        frame_idx: int,
        observation,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> SubgoalExecutionResult:
        update = self._plan_command(
            frame_idx=frame_idx,
            observation=observation,
            action_command=action_command,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
        )
        now = time.monotonic()
        if update.plan_version > self._last_applied_plan_version:
            if update.planner_control_mode in {None, "trajectory"}:
                self._tracker.set_trajectory(update.trajectory_world, plan_version=int(update.plan_version), timestamp=now)
                self._planner_yaw_target_rad = None
            else:
                self._tracker.clear(timestamp=now)
                if update.planner_control_mode == "yaw_delta" and update.planner_yaw_delta_rad is not None:
                    self._planner_yaw_target_rad = wrap_to_pi(float(robot_yaw) + float(update.planner_yaw_delta_rad))
                else:
                    self._planner_yaw_target_rad = None
            self._last_applied_plan_version = int(update.plan_version)

        evaluation = self.evaluate_action(
            action_command=action_command,
            trajectory_update=update,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )

        if action_command is not None and action_command.action_type == "LOOK_AT":
            self._command = self._look_at_command(action_command, float(robot_yaw))
        elif update.planner_control_mode == "yaw_delta":
            self._command = self._planner_yaw_command(float(robot_yaw))
        elif update.planner_control_mode in {"stop", "wait"}:
            self._command = np.zeros(3, dtype=np.float32)
        else:
            tracker_result = self._tracker.compute_command(
                np.asarray(robot_pos_world, dtype=np.float32),
                np.asarray(robot_quat_wxyz, dtype=np.float32),
                now=now,
                force_stop=evaluation.force_stop,
            )
            self._command = tracker_result.command

        status = self.build_status(
            action_command=action_command,
            trajectory_update=update,
            robot_pose=tuple(float(v) for v in np.asarray(robot_pos_world, dtype=np.float32).reshape(-1)[:3]),
            evaluation=evaluation,
        )
        return SubgoalExecutionResult(
            command_vector=self._command.copy(),
            trajectory_update=update,
            evaluation=evaluation,
            status=status,
        )

    def build_status(
        self,
        *,
        action_command: ActionCommand | None,
        trajectory_update: TrajectoryUpdate,
        robot_pose: tuple[float, float, float],
        evaluation: CommandEvaluation,
    ) -> ActionStatus | None:
        if action_command is None:
            return None
        planner_managed = bool(action_command.metadata.get("planner_managed", False))
        if action_command.action_type == "STOP":
            return ActionStatus(
                command_id=action_command.command_id,
                state="succeeded",
                success=True,
                robot_pose_xyz=robot_pose,
                distance_remaining_m=0.0,
                metadata={"action_type": action_command.action_type},
            )
        if evaluation.reached_goal:
            return ActionStatus(
                command_id=action_command.command_id,
                state="succeeded",
                success=True,
                robot_pose_xyz=robot_pose,
                distance_remaining_m=max(float(evaluation.goal_distance_m), 0.0),
                metadata={"action_type": action_command.action_type, "planner_managed": planner_managed},
            )
        if planner_managed and bool(trajectory_update.stop) and trajectory_update.stats.last_error == "":
            return ActionStatus(
                command_id=action_command.command_id,
                state="succeeded",
                success=True,
                robot_pose_xyz=robot_pose,
                distance_remaining_m=0.0,
                metadata={"action_type": action_command.action_type, "planner_managed": True},
            )
        if (
            trajectory_update.stats.last_error != ""
            and trajectory_update.trajectory_world.shape[0] == 0
            and action_command.action_type not in {"STOP", "LOOK_AT"}
        ):
            return ActionStatus(
                command_id=action_command.command_id,
                state="failed",
                success=False,
                reason=trajectory_update.stats.last_error,
                robot_pose_xyz=robot_pose,
                distance_remaining_m=None if evaluation.goal_distance_m < 0.0 else float(evaluation.goal_distance_m),
                metadata={"action_type": action_command.action_type, "planner_managed": planner_managed},
            )
        if action_command.action_type == "LOOK_AT" and abs(float(evaluation.yaw_error_rad)) < 0.05:
            return ActionStatus(
                command_id=action_command.command_id,
                state="succeeded",
                success=True,
                robot_pose_xyz=robot_pose,
                distance_remaining_m=0.0,
                metadata={"action_type": action_command.action_type},
            )
        return ActionStatus(
            command_id=action_command.command_id,
            state="running",
            success=False,
            robot_pose_xyz=robot_pose,
            distance_remaining_m=None if evaluation.goal_distance_m < 0.0 else float(evaluation.goal_distance_m),
            metadata={"action_type": action_command.action_type, "planner_managed": planner_managed},
        )

    def evaluate_action(
        self,
        *,
        action_command: ActionCommand | None,
        trajectory_update: TrajectoryUpdate | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> CommandEvaluation:
        if action_command is None:
            return CommandEvaluation(force_stop=True, goal_distance_m=-1.0, yaw_error_rad=0.0, reached_goal=False)
        if bool(action_command.metadata.get("planner_managed", False)):
            planner_mode = trajectory_update.planner_control_mode if trajectory_update is not None else None
            if planner_mode == "yaw_delta":
                yaw_target = float(self._planner_yaw_target_rad if self._planner_yaw_target_rad is not None else robot_yaw)
                yaw_error = wrap_to_pi(yaw_target - robot_yaw)
                reached_goal = abs(float(yaw_error)) < 0.05
                return CommandEvaluation(
                    force_stop=bool(reached_goal),
                    goal_distance_m=-1.0,
                    yaw_error_rad=float(yaw_error),
                    reached_goal=bool(reached_goal),
                )
            if planner_mode == "wait":
                return CommandEvaluation(force_stop=True, goal_distance_m=-1.0, yaw_error_rad=0.0, reached_goal=False)
            planner_stop = bool(trajectory_update.stop) if trajectory_update is not None else False
            return CommandEvaluation(force_stop=planner_stop, goal_distance_m=-1.0, yaw_error_rad=0.0, reached_goal=planner_stop)
        if action_command.action_type == "STOP":
            return CommandEvaluation(force_stop=True, goal_distance_m=0.0, yaw_error_rad=0.0, reached_goal=False)
        if action_command.action_type == "LOOK_AT":
            yaw_target = float(action_command.look_at_yaw_rad if action_command.look_at_yaw_rad is not None else robot_yaw)
            yaw_error = wrap_to_pi(yaw_target - robot_yaw)
            return CommandEvaluation(
                force_stop=abs(float(yaw_error)) < 0.05,
                goal_distance_m=0.0,
                yaw_error_rad=float(yaw_error),
                reached_goal=False,
            )
        if action_command.target_pose_xyz is None:
            return CommandEvaluation(force_stop=False, goal_distance_m=-1.0, yaw_error_rad=0.0, reached_goal=False)
        reached_goal = False
        goal_distance = xy_distance(np.asarray(robot_pos_world, dtype=np.float32), np.asarray(action_command.target_pose_xyz, dtype=np.float32))
        if action_command.action_type in {"NAV_TO_POSE", "NAV_TO_PLACE"}:
            reached_goal = within_xy_radius(
                np.asarray(robot_pos_world, dtype=np.float32),
                np.asarray(action_command.target_pose_xyz, dtype=np.float32),
                float(action_command.stop_radius_m),
            )
        return CommandEvaluation(
            force_stop=bool(reached_goal),
            goal_distance_m=float(goal_distance),
            yaw_error_rad=0.0,
            reached_goal=bool(reached_goal),
        )

    def empty_update(
        self,
        *,
        frame_idx: int,
        action_command: ActionCommand | None,
        error: str = "",
        stop: bool = False,
    ) -> TrajectoryUpdate:
        failed_calls = 1 if error != "" else 0
        return TrajectoryUpdate(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=self._last_applied_plan_version,
            stats=PlannerStats(failed_calls=failed_calls, last_error=error, last_plan_step=int(frame_idx)),
            source_frame_id=int(frame_idx),
            action_command=action_command,
            stop=bool(stop),
        )

    def _plan_command(
        self,
        *,
        frame_idx: int,
        observation,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> TrajectoryUpdate:
        if observation is None:
            return self.empty_update(frame_idx=frame_idx, action_command=action_command, error="sensor data unavailable", stop=True)
        if action_command is not None and action_command.action_type == "LOOK_AT":
            return self.empty_update(frame_idx=frame_idx, action_command=action_command, stop=False)
        return self._planning_session.plan_with_observation(
            observation,
            action_command=action_command,
            robot_pos_world=robot_pos_world,
            robot_yaw=float(robot_yaw),
            robot_quat_wxyz=robot_quat_wxyz,
        )

    def _look_at_command(self, action_command: ActionCommand, robot_yaw: float) -> np.ndarray:
        yaw_target = float(action_command.look_at_yaw_rad if action_command.look_at_yaw_rad is not None else robot_yaw)
        yaw_error = wrap_to_pi(yaw_target - robot_yaw)
        wz = np.clip(1.5 * float(yaw_error), -float(self.args.cmd_max_wz), float(self.args.cmd_max_wz))
        return np.asarray([0.0, 0.0, float(wz)], dtype=np.float32)

    def _planner_yaw_command(self, robot_yaw: float) -> np.ndarray:
        yaw_target = float(self._planner_yaw_target_rad if self._planner_yaw_target_rad is not None else robot_yaw)
        yaw_error = wrap_to_pi(yaw_target - robot_yaw)
        if abs(float(yaw_error)) < 0.05:
            return np.zeros(3, dtype=np.float32)
        wz = np.clip(1.5 * float(yaw_error), -float(self.args.cmd_max_wz), float(self.args.cmd_max_wz))
        return np.asarray([0.0, 0.0, float(wz)], dtype=np.float32)
