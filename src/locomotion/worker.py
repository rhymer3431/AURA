from __future__ import annotations

import time

import numpy as np

from common.geometry import within_xy_radius, wrap_to_pi, xy_distance
from control.trajectory_tracker import TrajectoryTracker, TrajectoryTrackerConfig
from ipc.messages import ActionCommand
from locomotion.types import CommandEvaluation
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate
from schemas.commands import LocomotionProposal

try:
    from control.navdp_follower import NavDPFollower, NavDPFollowerConfig

    _FOLLOWER_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    NavDPFollower = None
    NavDPFollowerConfig = None
    _FOLLOWER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class LocomotionWorker:
    def __init__(self, args, *, follower=None) -> None:
        self.args = args
        self._use_navdp_follower = bool(getattr(args, "use_navdp_follower", False))
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
                handoff_reset_distance_m=float(getattr(args, "traj_handoff_reset_distance_m", 0.35)),
                handoff_reset_heading_rad=float(getattr(args, "traj_handoff_reset_heading_rad", 0.5)),
            )
        )
        self._last_applied_plan_version = -1
        self._last_goal_version = -1
        self._last_planner_control_version = -1
        self._planner_yaw_target_rad: float | None = None
        self._planner_action_mode: str | None = None
        self._planner_action_started_at_s: float = 0.0
        self._planner_action_start_pos_xy: np.ndarray | None = None
        self._planner_action_start_yaw_rad: float = 0.0
        self._command = np.zeros(3, dtype=np.float32)
        self._follower = follower if self._use_navdp_follower else None
        self._follower_init_error = ""

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        del simulation_app, stage

    def shutdown(self) -> None:
        close = getattr(self._follower, "close", None)
        if callable(close):
            close()
        return None

    def command(self) -> np.ndarray:
        return self._command.copy()

    def execute(
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
    ) -> LocomotionProposal:
        del frame_idx, observation
        update = trajectory_update
        now = time.monotonic()
        planner_control_version = int(getattr(update, "planner_control_version", -1))
        if update.plan_version > self._last_applied_plan_version:
            if update.planner_control_mode in {None, "trajectory"}:
                reset_progress = True
                seed_progress_idx = None
                goal_version = int(getattr(update, "goal_version", -1))
                if goal_version >= 0 and goal_version == self._last_goal_version:
                    handoff = self._tracker.compute_handoff(
                        update.trajectory_world,
                        position_w=np.asarray(robot_pos_world, dtype=np.float32),
                    )
                    reset_progress = bool(handoff.reset_progress)
                    seed_progress_idx = handoff.seed_progress_idx
                self._tracker.set_trajectory(
                    update.trajectory_world,
                    plan_version=int(update.plan_version),
                    timestamp=now,
                    reset_progress=reset_progress,
                    seed_progress_idx=seed_progress_idx,
                )
                self._planner_yaw_target_rad = None
                self._planner_action_mode = None
                self._planner_action_start_pos_xy = None
            else:
                self._apply_direct_planner_control(
                    update=update,
                    now=now,
                    robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                    robot_yaw=float(robot_yaw),
                )
            self._last_applied_plan_version = int(update.plan_version)
            self._last_goal_version = int(getattr(update, "goal_version", -1))
            self._last_planner_control_version = max(self._last_planner_control_version, planner_control_version)
        elif (
            update.planner_control_mode not in {None, "trajectory"}
            and planner_control_version > self._last_planner_control_version
        ):
            self._apply_direct_planner_control(
                update=update,
                now=now,
                robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(robot_yaw),
            )
            self._last_planner_control_version = planner_control_version

        evaluation = self.evaluate_action(
            action_command=action_command,
            trajectory_update=update,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )
        metadata = self._planner_metadata(update)
        state_label = self._state_label_for_update(update)

        if action_command is not None and action_command.action_type == "LOOK_AT":
            self._command = self._look_at_command(action_command, float(robot_yaw))
            state_label = "look-at"
        elif update.planner_control_mode == "yaw_delta":
            self._command = self._planner_yaw_command(float(robot_yaw))
            state_label = "yaw-delta"
        elif update.planner_control_mode in {"forward", "yaw_left", "yaw_right"}:
            self._command = self._planner_action_command(
                planner_mode=str(update.planner_control_mode),
                robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(robot_yaw),
                now=now,
            )
            if update.planner_control_mode == "forward":
                state_label = "forward-override"
            elif update.planner_control_mode == "yaw_left":
                state_label = "yaw-left-override"
            else:
                state_label = "yaw-right-override"
        elif update.planner_control_mode in {"stop", "wait"}:
            self._command = np.zeros(3, dtype=np.float32)
            state_label = "waiting"
        else:
            self._command, trajectory_metadata = self._trajectory_command(
                robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                robot_lin_vel_world=np.asarray(robot_lin_vel_world, dtype=np.float32),
                robot_ang_vel_world=np.asarray(robot_ang_vel_world, dtype=np.float32),
                robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
                now=now,
                evaluation=evaluation,
                stale_hold_reason=str(getattr(update, "stale_hold_reason", "")),
            )
            metadata.update(trajectory_metadata)
            state_label = self._state_label_for_update(update)
        metadata["locomotion_state_label"] = state_label
        return LocomotionProposal(
            command_vector=self._command.copy(),
            trajectory_update=update,
            evaluation=evaluation,
            metadata=metadata,
        )

    @staticmethod
    def _planner_metadata(update: TrajectoryUpdate) -> dict[str, object]:
        return {
            "planner_control_mode": None if update.planner_control_mode is None else str(update.planner_control_mode),
            "planner_control_queue": [str(item) for item in getattr(update, "planner_control_queue", ())],
            "planner_control_progress": float(getattr(update, "planner_control_progress", 0.0) or 0.0),
            "stale_hold_reason": str(getattr(update, "stale_hold_reason", "")),
        }

    @staticmethod
    def _state_label_for_update(update: TrajectoryUpdate) -> str:
        mode = None if update.planner_control_mode is None else str(update.planner_control_mode)
        if mode == "forward":
            return "forward-override"
        if mode == "yaw_left":
            return "yaw-left-override"
        if mode == "yaw_right":
            return "yaw-right-override"
        if mode == "stop":
            return "waiting"
        if mode == "wait":
            if str(getattr(update, "planner_control_reason", "")) == "goal_reached":
                return "done"
            return "waiting"
        stale_hold_reason = str(getattr(update, "stale_hold_reason", ""))
        if stale_hold_reason == "stale_hold":
            return "stale-hold"
        if update.trajectory_world.shape[0] == 0:
            return "tracking"
        return "tracking"

    def _apply_direct_planner_control(
        self,
        *,
        update: TrajectoryUpdate,
        now: float,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> None:
        self._tracker.clear(timestamp=now)
        if update.planner_control_mode == "yaw_delta" and update.planner_yaw_delta_rad is not None:
            self._planner_yaw_target_rad = wrap_to_pi(float(robot_yaw) + float(update.planner_yaw_delta_rad))
            self._planner_action_mode = None
            self._planner_action_start_pos_xy = None
            return
        if update.planner_control_mode in {"forward", "yaw_left", "yaw_right"}:
            self._planner_yaw_target_rad = None
            self._planner_action_mode = str(update.planner_control_mode)
            self._planner_action_started_at_s = float(now)
            self._planner_action_start_pos_xy = np.asarray(robot_pos_world, dtype=np.float32).reshape(-1)[:2].copy()
            self._planner_action_start_yaw_rad = float(robot_yaw)
            return
        self._planner_yaw_target_rad = None
        self._planner_action_mode = None
        self._planner_action_start_pos_xy = None

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
            if planner_mode in {"forward", "yaw_left", "yaw_right"}:
                return CommandEvaluation(force_stop=False, goal_distance_m=-1.0, yaw_error_rad=0.0, reached_goal=False)
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

    def _planner_action_command(
        self,
        *,
        planner_mode: str,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        now: float,
    ) -> np.ndarray:
        timeout_s = max(0.1, float(getattr(self.args, "internvla_action_timeout_s", 1.5)))
        elapsed = float(now - self._planner_action_started_at_s)
        if elapsed > timeout_s:
            return np.zeros(3, dtype=np.float32)
        if planner_mode == "forward":
            start_xy = self._planner_action_start_pos_xy
            if start_xy is None:
                return np.zeros(3, dtype=np.float32)
            target_distance = max(0.05, float(getattr(self.args, "internvla_forward_step_m", 0.25)))
            distance = xy_distance(np.asarray(start_xy, dtype=np.float32), np.asarray(robot_pos_world, dtype=np.float32))
            if distance >= target_distance:
                return np.zeros(3, dtype=np.float32)
            vx = min(float(self.args.cmd_max_vx), max(0.05, float(getattr(self.args, "obstacle_slow_forward_vx_mps", 0.08))))
            return np.asarray([float(vx), 0.0, 0.0], dtype=np.float32)
        if planner_mode in {"yaw_left", "yaw_right"}:
            target_yaw = max(0.05, float(np.deg2rad(float(getattr(self.args, "internvla_turn_step_deg", 20.0)))))
            yaw_delta = abs(float(wrap_to_pi(robot_yaw - float(self._planner_action_start_yaw_rad))))
            if yaw_delta >= target_yaw:
                return np.zeros(3, dtype=np.float32)
            yaw_sign = 1.0 if planner_mode == "yaw_left" else -1.0
            wz = min(float(self.args.cmd_max_wz), 0.5) * yaw_sign
            return np.asarray([0.0, 0.0, float(wz)], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _trajectory_command(
        self,
        *,
        robot_pos_world: np.ndarray,
        robot_lin_vel_world: np.ndarray,
        robot_ang_vel_world: np.ndarray,
        robot_quat_wxyz: np.ndarray,
        now: float,
        evaluation: CommandEvaluation,
        stale_hold_reason: str,
    ) -> tuple[np.ndarray, dict[str, object]]:
        pose_target = self._tracker.compute_target_pose(
            robot_pos_world,
            robot_quat_wxyz,
            now=now,
            force_stop=evaluation.force_stop,
        )
        follower, init_error = self._ensure_follower()
        if follower is None:
            return self._legacy_tracker_command(
                robot_pos_world=robot_pos_world,
                robot_quat_wxyz=robot_quat_wxyz,
                now=now,
                evaluation=evaluation,
                fallback_reason=init_error or "navdp_follower_unavailable",
                stale_hold_reason=stale_hold_reason,
            )
        if pose_target.stale or evaluation.force_stop:
            return np.zeros(3, dtype=np.float32), {
                "trajectory_command_source": "navdp_follower",
                "stale_hold_reason": stale_hold_reason,
            }
        try:
            result = follower.compute_command(
                pose_command_b=pose_target.pose_command_b,
                base_lin_vel_w=robot_lin_vel_world,
                base_ang_vel_w=robot_ang_vel_world,
                robot_quat_wxyz=robot_quat_wxyz,
            )
        except Exception as exc:  # noqa: BLE001
            return self._legacy_tracker_command(
                robot_pos_world=robot_pos_world,
                robot_quat_wxyz=robot_quat_wxyz,
                now=now,
                evaluation=evaluation,
                fallback_reason=f"{type(exc).__name__}: {exc}",
                stale_hold_reason=stale_hold_reason,
            )
        return result.command, {
            "trajectory_command_source": "navdp_follower",
            "stale_hold_reason": stale_hold_reason,
        }

    def _legacy_tracker_command(
        self,
        *,
        robot_pos_world: np.ndarray,
        robot_quat_wxyz: np.ndarray,
        now: float,
        evaluation: CommandEvaluation,
        fallback_reason: str,
        stale_hold_reason: str,
    ) -> tuple[np.ndarray, dict[str, object]]:
        tracker_result = self._tracker.compute_command(
            robot_pos_world,
            robot_quat_wxyz,
            now=now,
            force_stop=evaluation.force_stop,
        )
        metadata = {
            "trajectory_command_source": "legacy_tracker",
            "stale_hold_reason": stale_hold_reason,
        }
        if str(fallback_reason).strip() != "":
            metadata["trajectory_fallback_reason"] = str(fallback_reason)
        return tracker_result.command, metadata

    def _ensure_follower(self):
        if not self._use_navdp_follower:
            return None, "navdp_follower_disabled"
        if self._follower is not None:
            return self._follower, ""
        if self._follower_init_error != "":
            return None, self._follower_init_error
        if NavDPFollower is None or NavDPFollowerConfig is None:
            self._follower_init_error = _FOLLOWER_IMPORT_ERROR or "navdp_follower_import_failed"
            return None, self._follower_init_error
        try:
            self._follower = NavDPFollower(
                NavDPFollowerConfig(
                    policy_path="",
                    onnx_device=str(getattr(self.args, "onnx_device", "auto")).strip().lower() or "auto",
                    max_vx=float(self.args.cmd_max_vx),
                    max_vy=float(self.args.cmd_max_vy),
                    max_wz=float(self.args.cmd_max_wz),
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._follower_init_error = f"{type(exc).__name__}: {exc}"
            return None, self._follower_init_error
        return self._follower, ""
