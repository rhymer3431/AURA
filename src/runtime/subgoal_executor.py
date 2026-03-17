from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

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
class ObstacleDefenseConfig:
    enabled: bool = True
    stop_distance_m: float = 0.45
    hold_distance_m: float = 0.70
    side_bias_m: float = 0.10
    min_valid_fraction: float = 0.05
    min_turn_wz: float = 0.35
    forward_trigger_mps: float = 0.05
    slow_forward_vx_mps: float = 0.08
    backoff_vx_mps: float = 0.18
    lateral_nudge_vy_mps: float = 0.12
    recovery_hold_sec: float = 0.75


@dataclass(frozen=True)
class ObstacleDefenseResult:
    command: np.ndarray
    triggered: bool
    metadata: dict[str, object]


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
        follower: Any | None = None,
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
        # Deprecated compatibility slot: older callers may still pass a follower,
        # but the runtime has been restored to direct tracker-owned command output.
        del follower
        self._obstacle_defense = ObstacleDefenseConfig(
            enabled=bool(getattr(args, "obstacle_defense_enabled", True)),
            stop_distance_m=float(getattr(args, "obstacle_stop_distance_m", 0.45)),
            hold_distance_m=float(getattr(args, "obstacle_hold_distance_m", getattr(args, "obstacle_turn_distance_m", 0.70))),
            side_bias_m=float(getattr(args, "obstacle_side_bias_m", 0.10)),
            min_valid_fraction=float(getattr(args, "obstacle_min_valid_fraction", 0.05)),
            min_turn_wz=float(getattr(args, "obstacle_min_turn_wz", 0.35)),
            forward_trigger_mps=float(getattr(args, "obstacle_forward_trigger_mps", 0.05)),
            slow_forward_vx_mps=float(getattr(args, "obstacle_slow_forward_vx_mps", 0.08)),
            backoff_vx_mps=float(getattr(args, "obstacle_backoff_vx_mps", 0.18)),
            lateral_nudge_vy_mps=float(getattr(args, "obstacle_lateral_nudge_vy_mps", 0.12)),
            recovery_hold_sec=float(getattr(args, "obstacle_recovery_hold_sec", 0.75)),
        )
        self._last_applied_plan_version = -1
        self._planner_yaw_target_rad: float | None = None
        self._command = np.zeros(3, dtype=np.float32)
        self._obstacle_recovery_until = 0.0
        self._obstacle_recovery_sign = 0

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
        robot_lin_vel_world: np.ndarray,
        robot_ang_vel_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> SubgoalExecutionResult:
        del robot_lin_vel_world, robot_ang_vel_world
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
        defense = self._apply_obstacle_defense(observation, self._command, now=now)
        self._command = defense.command

        status = self.build_status(
            action_command=action_command,
            trajectory_update=update,
            robot_pose=tuple(float(v) for v in np.asarray(robot_pos_world, dtype=np.float32).reshape(-1)[:3]),
            evaluation=evaluation,
        )
        if status is not None and defense.triggered:
            status = replace(status, metadata={**dict(status.metadata), **defense.metadata})
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

    def _apply_obstacle_defense(self, observation, command: np.ndarray, *, now: float | None = None) -> ObstacleDefenseResult:  # noqa: ANN001
        current_time = time.monotonic() if now is None else float(now)
        base_command = np.asarray(command, dtype=np.float32).copy()
        if not bool(self._obstacle_defense.enabled) or observation is None:
            return ObstacleDefenseResult(command=base_command, triggered=False, metadata={})

        depth = np.asarray(getattr(observation, "depth", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2 or depth.size == 0:
            return self._maybe_build_recovery_command(base_command, now=current_time)

        row_start = min(max(int(depth.shape[0] * 0.35), 0), depth.shape[0] - 1)
        row_stop = min(max(int(depth.shape[0] * 0.90), row_start + 1), depth.shape[0])
        col_start = min(max(int(depth.shape[1] * 0.20), 0), depth.shape[1] - 1)
        col_stop = min(max(int(depth.shape[1] * 0.80), col_start + 1), depth.shape[1])
        roi = depth[row_start:row_stop, col_start:col_stop]
        if roi.size == 0:
            return self._maybe_build_recovery_command(base_command, now=current_time)

        valid_mask = np.isfinite(roi) & (roi > 0.05)
        valid_count = int(np.count_nonzero(valid_mask))
        min_valid_count = max(int(roi.size * float(self._obstacle_defense.min_valid_fraction)), 4)
        if valid_count < min_valid_count:
            return self._maybe_build_recovery_command(base_command, now=current_time)

        left_roi, center_roi, right_roi = np.array_split(roi, 3, axis=1)
        left_clearance = self._depth_band_clearance(left_roi)
        center_clearance = self._depth_band_clearance(center_roi)
        right_clearance = self._depth_band_clearance(right_roi)

        moving_forward = float(base_command[0]) > float(self._obstacle_defense.forward_trigger_mps)
        turn_sign = self._clearer_side_turn_sign(left_clearance, right_clearance)
        metadata = {
            "obstacle_defense": True,
            "obstacle_left_clearance_m": round(left_clearance, 4),
            "obstacle_center_clearance_m": round(center_clearance, 4),
            "obstacle_right_clearance_m": round(right_clearance, 4),
            "obstacle_stop_distance_m": round(float(self._obstacle_defense.stop_distance_m), 4),
            "obstacle_hold_distance_m": round(float(self._obstacle_defense.hold_distance_m), 4),
        }

        if float(center_clearance) < float(self._obstacle_defense.stop_distance_m):
            preferred_sign = turn_sign if turn_sign != 0 else self._obstacle_recovery_sign
            self._activate_obstacle_recovery(current_time, preferred_sign)
            return self._build_backoff_command(
                base_command,
                turn_sign=preferred_sign,
                metadata=metadata,
                mode="backoff_turn" if preferred_sign != 0 else "backoff_hold",
                recovery_active=False,
            )

        self._clear_obstacle_recovery()

        if not moving_forward and float(center_clearance) >= float(self._obstacle_defense.stop_distance_m):
            self._clear_obstacle_recovery()
            return ObstacleDefenseResult(command=base_command, triggered=False, metadata={})

        if moving_forward and float(center_clearance) < float(self._obstacle_defense.hold_distance_m):
            self._clear_obstacle_recovery()
            metadata["obstacle_defense_mode"] = "hold"
            return ObstacleDefenseResult(
                command=np.zeros(3, dtype=np.float32),
                triggered=True,
                metadata=metadata,
            )

        self._clear_obstacle_recovery()
        return ObstacleDefenseResult(command=base_command, triggered=False, metadata={})

    @staticmethod
    def _depth_band_clearance(depth_band: np.ndarray) -> float:
        band = np.asarray(depth_band, dtype=np.float32)
        if band.ndim != 2 or band.size == 0:
            return float("inf")
        valid = band[np.isfinite(band) & (band > 0.05)]
        if valid.size == 0:
            return float("inf")
        return float(np.percentile(valid, 20.0))

    def _clearer_side_turn_sign(self, left_clearance: float, right_clearance: float) -> int:
        bias = float(self._obstacle_defense.side_bias_m)
        if right_clearance > left_clearance + bias:
            return -1
        if left_clearance > right_clearance + bias:
            return 1
        return 0

    def _activate_obstacle_recovery(self, now: float, turn_sign: int) -> None:
        self._obstacle_recovery_until = float(now) + float(self._obstacle_defense.recovery_hold_sec)
        self._obstacle_recovery_sign = int(turn_sign)

    def _clear_obstacle_recovery(self) -> None:
        self._obstacle_recovery_until = 0.0
        self._obstacle_recovery_sign = 0

    def _maybe_build_recovery_command(self, base_command: np.ndarray, *, now: float) -> ObstacleDefenseResult:
        if float(now) >= float(self._obstacle_recovery_until):
            return ObstacleDefenseResult(command=base_command, triggered=False, metadata={})
        return self._build_backoff_command(
            base_command,
            turn_sign=self._obstacle_recovery_sign,
            metadata={
                "obstacle_defense": True,
                "obstacle_depth_valid": False,
            },
            mode="backoff_recovery",
            recovery_active=True,
        )

    def _build_backoff_command(
        self,
        base_command: np.ndarray,
        *,
        turn_sign: int,
        metadata: dict[str, object],
        mode: str,
        recovery_active: bool,
    ) -> ObstacleDefenseResult:
        max_vx = max(float(getattr(self.args, "cmd_max_vx", 0.0)), 0.0)
        max_vy = max(float(getattr(self.args, "cmd_max_vy", 0.0)), 0.0)
        max_wz = max(float(getattr(self.args, "cmd_max_wz", 0.0)), 1.0e-3)
        backoff_vx = -min(float(self._obstacle_defense.backoff_vx_mps), max_vx)
        lateral_vy = 0.0
        turn_wz = 0.0
        if turn_sign != 0:
            lateral_vy = float(turn_sign) * min(float(self._obstacle_defense.lateral_nudge_vy_mps), max_vy)
            turn_wz = float(turn_sign) * min(max_wz, max(float(self._obstacle_defense.min_turn_wz), abs(float(base_command[2]))))
        payload = {
            **metadata,
            "obstacle_defense_mode": mode,
        }
        if recovery_active:
            payload["obstacle_recovery_active"] = True
        return ObstacleDefenseResult(
            command=np.asarray([backoff_vx, lateral_vy, turn_wz], dtype=np.float32),
            triggered=True,
            metadata=payload,
        )
