from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from common.geometry import within_xy_radius, world_goal_to_robot_frame
from control.async_planners import (
    NoGoalPlannerInput,
    PlannerInput,
    PlannerOutput,
)
from inference.vlm import AsyncSystem2Input, System2Result, resolve_goal_world_xy_from_pixel
from ipc.messages import ActionCommand
from runtime.global_route import GlobalRoutePlanner
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate
from schemas.execution_mode import normalize_execution_mode

from .planner_runtime_state import PlannerRuntimeState


class PlannerRuntimeEngine:
    def __init__(self, args, *, transport, state: PlannerRuntimeState) -> None:  # noqa: ANN001
        self._args = args
        self._transport = transport
        self._state = state

    @property
    def state(self) -> PlannerRuntimeState:
        return self._state

    def _interactive_enabled(self) -> bool:
        return str(getattr(self._args, "planner_mode", "")).strip().lower() == "interactive"

    def active_memory_instruction(self) -> str:
        return self._state.active_memory_instruction()

    def viewer_overlay_state(self) -> dict[str, object]:
        return self._state.viewer_overlay_state()

    def start_nav_task(self, instruction: str) -> None:
        if self._state.mode != "NAV":
            raise RuntimeError("start_nav_task requires execution mode NAV")
        text = str(instruction).strip()
        if text == "":
            raise ValueError("navigation instruction must be non-empty")
        if getattr(self._transport, "system2_client", None) is None or getattr(self._transport, "pointgoal_planner", None) is None:
            raise RuntimeError("NAV task planner is not initialized")
        self._transport.system2_client.reset(text)
        try:
            self._transport.navdp_client.navigator_reset(self._transport._intrinsic.copy(), batch_size=1)
        except Exception:
            pass
        self._state.goal.nav_instruction = text
        self._state.trajectory.plan_version = -1
        self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
        self._state.reset_planner_control()
        self._state.trajectory.stale_sec = -1.0
        self._state.goal.goal_version = -1
        self._state.goal.traj_version = -1
        self._state.goal.system2_result_version = -1
        self._state.goal.system2_pixel_goal = None
        self._state.goal.system2_submit_ts = 0.0
        self._state.goal.system2_response_ts = time.perf_counter()
        self._state.goal.local_xy = np.zeros(2, dtype=np.float32)
        self._state.interactive.last_nogoal_plan_version = -1
        self._state.interactive.last_nav_plan_version = -1
        self._state.trajectory.stats = PlannerStats()
        if getattr(self._transport, "system2_planner", None) is not None:
            self._transport.system2_planner.reset_state()

    def submit_interactive_instruction(self, instruction: str) -> int:
        if not self._interactive_enabled():
            raise RuntimeError("submit_interactive_instruction requires planner-mode=interactive")
        text = str(instruction).strip()
        if text == "":
            raise ValueError("interactive instruction must be non-empty")
        interactive = self._state.interactive
        with interactive.lock:
            interactive.command_seq += 1
            interactive.pending_command_id = int(interactive.command_seq)
            interactive.pending_instruction = text
            interactive.cancel_requested = False
            return int(interactive.pending_command_id)

    def cancel_interactive_task(self) -> bool:
        if not self._interactive_enabled():
            return False
        interactive = self._state.interactive
        with interactive.lock:
            has_work = (
                interactive.pending_command_id >= 0
                or interactive.active_command_id >= 0
                or interactive.phase in {"task_pending", "task_active"}
            )
            if not has_work:
                return False
            interactive.pending_command_id = -1
            interactive.pending_instruction = ""
            interactive.cancel_requested = True
            return True

    def activate_interactive_roaming(self, reason: str) -> bool:
        return self._activate_roaming(reason)

    def plan_with_observation(
        self,
        observation: ExecutionObservation,
        *,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> TrajectoryUpdate:
        del robot_quat_wxyz
        if self._interactive_enabled():
            return self._run_interactive(
                observation,
                action_command=action_command,
                robot_pos_world=robot_pos_world,
                robot_yaw=robot_yaw,
            )
        if self._state.mode == "NAV":
            if getattr(self._transport, "pointgoal_planner", None) is None or getattr(self._transport, "system2_client", None) is None:
                raise RuntimeError("PlanningSession is not initialized.")
            return self._run_nav_task(
                observation,
                action_command=action_command,
                robot_pos_world=robot_pos_world,
                robot_yaw=robot_yaw,
            )
        if getattr(self._transport, "pointgoal_planner", None) is None or getattr(self._transport, "nogoal_planner", None) is None:
            raise RuntimeError("PlanningSession is not initialized.")
        if action_command is None or action_command.action_type in {"STOP", "LOOK_AT"}:
            return TrajectoryUpdate(
                trajectory_world=np.zeros((0, 3), dtype=np.float32),
                plan_version=self._state.trajectory.plan_version,
                stats=PlannerStats(last_plan_step=int(observation.frame_id)),
                source_frame_id=int(observation.frame_id),
                action_command=action_command,
                stop=True,
                sensor_meta=dict(observation.sensor_meta),
            )
        if self._state.mode == "EXPLORE" or action_command.action_type == "LOCAL_SEARCH":
            return self._run_nogoal(observation, action_command=action_command)
        if action_command.target_pose_xyz is None:
            return TrajectoryUpdate(
                trajectory_world=np.asarray(self._state.trajectory.trajectory_world, dtype=np.float32).copy(),
                plan_version=self._state.trajectory.plan_version,
                stats=PlannerStats(
                    failed_calls=1,
                    last_error="target_pose_xyz is required",
                    last_plan_step=int(observation.frame_id),
                ),
                source_frame_id=int(observation.frame_id),
                action_command=action_command,
                stop=True,
                sensor_meta=dict(observation.sensor_meta),
            )
        goal_local_xy, route_error = self._resolve_pointgoal_local_goal(
            action_command=action_command,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )
        if route_error != "":
            return self._pointgoal_failure_update(
                frame_id=int(observation.frame_id),
                action_command=action_command,
                error=route_error,
                sensor_meta=observation.sensor_meta,
            )
        assert goal_local_xy is not None
        return self._run_pointgoal(
            observation,
            goal_local_xy,
            robot_pos_world,
            robot_yaw,
            action_command,
        )

    def update(
        self,
        frame_id: int,
        *,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
        env=None,  # noqa: ANN001
    ) -> TrajectoryUpdate:
        observation = self._transport.capture_observation(frame_id, env=env)
        if observation is None:
            return TrajectoryUpdate(
                trajectory_world=np.asarray(self._state.trajectory.trajectory_world, dtype=np.float32).copy(),
                plan_version=self._state.trajectory.plan_version,
                stats=PlannerStats(failed_calls=1, last_error="sensor data unavailable", last_plan_step=int(frame_id)),
                source_frame_id=int(frame_id),
                action_command=action_command,
                stop=True,
            )
        return self.plan_with_observation(
            observation,
            action_command=action_command,
            robot_pos_world=robot_pos_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )

    def _resolve_pointgoal_local_goal(
        self,
        *,
        action_command: ActionCommand,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> tuple[np.ndarray | None, str]:
        goal_xy = np.asarray(action_command.target_pose_xyz[:2], dtype=np.float32)
        robot_xy = np.asarray(robot_pos_world[:2], dtype=np.float32)
        if not self._global_route_enabled():
            self._state.clear_global_route_progress()
            goal_local_xy = world_goal_to_robot_frame(goal_xy=goal_xy, robot_xy=robot_xy, robot_yaw=float(robot_yaw))
            return goal_local_xy.astype(np.float32), ""
        try:
            waypoint_xy = self._resolve_global_route_waypoint(
                robot_xy=robot_xy,
                goal_xy=goal_xy,
                stop_radius_m=float(action_command.stop_radius_m),
            )
        except Exception as exc:  # noqa: BLE001
            error = f"global route planning failed: {type(exc).__name__}: {exc}"
            self._state.global_route.last_error = error
            return None, error
        goal_local_xy = world_goal_to_robot_frame(goal_xy=waypoint_xy, robot_xy=robot_xy, robot_yaw=float(robot_yaw))
        return goal_local_xy.astype(np.float32), ""

    def _pointgoal_failure_update(
        self,
        *,
        frame_id: int,
        action_command: ActionCommand,
        error: str,
        sensor_meta: dict[str, Any] | None = None,
    ) -> TrajectoryUpdate:
        self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
        self._state.goal.local_xy = np.zeros(2, dtype=np.float32)
        self._state.reset_planner_control()
        self._state.trajectory.stats = PlannerStats(
            failed_calls=1,
            last_error=str(error),
            last_plan_step=int(frame_id),
        )
        return TrajectoryUpdate(
            trajectory_world=np.asarray(self._state.trajectory.trajectory_world, dtype=np.float32).copy(),
            plan_version=int(self._state.trajectory.plan_version),
            stats=self._state.trajectory.stats,
            source_frame_id=int(frame_id),
            goal_local_xy=np.asarray(self._state.goal.local_xy, dtype=np.float32).copy(),
            action_command=action_command,
            stop=True,
            sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
        )

    def _global_route_enabled(self) -> bool:
        return self._state.mode == "MEM_NAV" and str(getattr(self._args, "global_map_image", "")).strip() != ""

    def _ensure_global_route_planner(self) -> GlobalRoutePlanner:
        route_state = self._state.global_route
        if route_state.planner is not None:
            return route_state.planner
        map_image_path = Path(str(getattr(self._args, "global_map_image", "")).strip())
        if str(map_image_path) == "":
            raise RuntimeError("global route map image is not configured")
        config_override = str(getattr(self._args, "global_map_config", "")).strip()
        config_path = Path(config_override) if config_override != "" else map_image_path.parent / "config.txt"
        planner = GlobalRoutePlanner.from_files(
            image_path=map_image_path,
            config_path=config_path,
            inflation_radius_m=float(getattr(self._args, "global_inflation_radius_m", 0.25)),
        )
        route_state.planner = planner
        return planner

    def _resolve_global_route_waypoint(
        self,
        *,
        robot_xy: np.ndarray,
        goal_xy: np.ndarray,
        stop_radius_m: float,
    ) -> np.ndarray:
        if within_xy_radius(
            np.asarray([float(robot_xy[0]), float(robot_xy[1]), 0.0], dtype=np.float32),
            np.asarray([float(goal_xy[0]), float(goal_xy[1]), 0.0], dtype=np.float32),
            float(stop_radius_m),
        ):
            return goal_xy.astype(np.float32).copy()
        self._ensure_global_route_planner()
        route_state = self._state.global_route
        replan_reason = self._global_route_replan_reason(robot_xy=robot_xy, goal_xy=goal_xy, stop_radius_m=stop_radius_m)
        if replan_reason != "":
            self._replan_global_route(robot_xy=robot_xy, goal_xy=goal_xy, reason=replan_reason)
        self._advance_global_route_index(robot_xy=robot_xy)
        if route_state.active_index >= len(route_state.waypoints_world):
            if within_xy_radius(
                np.asarray([float(robot_xy[0]), float(robot_xy[1]), 0.0], dtype=np.float32),
                np.asarray([float(goal_xy[0]), float(goal_xy[1]), 0.0], dtype=np.float32),
                float(stop_radius_m),
            ):
                return goal_xy.astype(np.float32).copy()
            self._replan_global_route(robot_xy=robot_xy, goal_xy=goal_xy, reason="route_exhausted_before_goal")
            self._advance_global_route_index(robot_xy=robot_xy)
        if route_state.active_index >= len(route_state.waypoints_world):
            raise RuntimeError("global route has no active waypoint after replanning")
        active_waypoint = np.asarray(route_state.waypoints_world[route_state.active_index], dtype=np.float32)
        distance = float(np.linalg.norm(active_waypoint - robot_xy[:2]))
        if not np.isfinite(route_state.best_distance_m) or distance <= route_state.best_distance_m - 0.10:
            route_state.best_distance_m = distance
            route_state.last_progress_ts = time.monotonic()
        return active_waypoint

    def _global_route_replan_reason(
        self,
        *,
        robot_xy: np.ndarray,
        goal_xy: np.ndarray,
        stop_radius_m: float,
    ) -> str:
        route_state = self._state.global_route
        if len(route_state.waypoints_world) == 0:
            return "route_missing"
        if not np.allclose(route_state.final_goal_xy[:2], goal_xy[:2], atol=1.0e-4):
            return "goal_changed"
        self._advance_global_route_index(robot_xy=robot_xy)
        if route_state.active_index >= len(route_state.waypoints_world):
            if within_xy_radius(
                np.asarray([float(robot_xy[0]), float(robot_xy[1]), 0.0], dtype=np.float32),
                np.asarray([float(goal_xy[0]), float(goal_xy[1]), 0.0], dtype=np.float32),
                float(stop_radius_m),
            ):
                return ""
            return "route_exhausted_before_goal"
        active_waypoint = np.asarray(route_state.waypoints_world[route_state.active_index], dtype=np.float32)
        distance = float(np.linalg.norm(active_waypoint - robot_xy[:2]))
        if not np.isfinite(route_state.best_distance_m):
            route_state.best_distance_m = distance
            route_state.last_progress_ts = time.monotonic()
            return ""
        if distance <= route_state.best_distance_m - 0.10:
            route_state.best_distance_m = distance
            route_state.last_progress_ts = time.monotonic()
            return ""
        if time.monotonic() - float(route_state.last_progress_ts) >= 2.0 and distance > 0.35:
            return "stuck_no_progress"
        return ""

    def _advance_global_route_index(self, *, robot_xy: np.ndarray) -> None:
        route_state = self._state.global_route
        now = time.monotonic()
        while route_state.active_index < len(route_state.waypoints_world):
            active_waypoint = np.asarray(route_state.waypoints_world[route_state.active_index], dtype=np.float32)
            if float(np.linalg.norm(active_waypoint - robot_xy[:2])) >= 0.35:
                break
            route_state.active_index += 1
            route_state.best_distance_m = float("inf")
            route_state.last_progress_ts = now

    def _replan_global_route(self, *, robot_xy: np.ndarray, goal_xy: np.ndarray, reason: str) -> None:
        planner = self._ensure_global_route_planner()
        route_state = self._state.global_route
        route_state.last_replan_reason = str(reason)
        route_state.last_error = ""
        waypoints = planner.plan(
            start_xy=(float(robot_xy[0]), float(robot_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            waypoint_spacing_m=float(getattr(self._args, "global_waypoint_spacing_m", 0.75)),
        )
        route_state.final_goal_xy = goal_xy.astype(np.float32).copy()
        route_state.waypoints_world = [(float(x), float(y)) for x, y in waypoints]
        route_state.active_index = 0
        route_state.last_progress_ts = time.monotonic()
        route_state.best_distance_m = float("inf")
        route_state.last_error = ""

    def _run_pointgoal(
        self,
        observation: ExecutionObservation,
        goal_local_xy: np.ndarray,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        action_command: ActionCommand,
    ) -> TrajectoryUpdate:
        self._state.goal.local_xy = np.asarray(goal_local_xy, dtype=np.float32).copy()
        self._transport.pointgoal_planner.submit(
            PlannerInput(
                frame_id=observation.frame_id,
                local_goal_xy=np.asarray(goal_local_xy, dtype=np.float32),
                rgb=observation.rgb,
                depth=observation.depth,
                sensor_meta=observation.sensor_meta,
                cam_pos=observation.cam_pos,
                cam_quat=observation.cam_quat,
                robot_pos=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(robot_yaw),
            )
        )
        return self._consume_planner_latest(
            planner=self._transport.pointgoal_planner,
            frame_id=observation.frame_id,
            action_command=action_command,
            goal_local_xy=np.asarray(goal_local_xy, dtype=np.float32),
            sensor_meta=observation.sensor_meta,
        )

    def _run_nogoal(self, observation: ExecutionObservation, *, action_command: ActionCommand | None) -> TrajectoryUpdate:
        self._transport.nogoal_planner.submit(
            NoGoalPlannerInput(
                frame_id=observation.frame_id,
                rgb=observation.rgb,
                depth=observation.depth,
                sensor_meta=observation.sensor_meta,
                cam_pos=observation.cam_pos,
                cam_quat=observation.cam_quat,
            )
        )
        return self._consume_planner_latest(
            planner=self._transport.nogoal_planner,
            frame_id=observation.frame_id,
            action_command=action_command,
            sensor_meta=observation.sensor_meta,
        )

    def _run_nav_task(
        self,
        observation: ExecutionObservation,
        *,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> TrajectoryUpdate:
        if self._state.goal.nav_instruction == "":
            return TrajectoryUpdate(
                trajectory_world=np.zeros((0, 3), dtype=np.float32),
                plan_version=self._state.trajectory.plan_version,
                stats=PlannerStats(failed_calls=1, last_error="NAV task not started", last_plan_step=int(observation.frame_id)),
                source_frame_id=int(observation.frame_id),
                action_command=action_command,
                stop=True,
                sensor_meta=dict(observation.sensor_meta),
            )
        self._maybe_update_system2(
            observation,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )
        self._maybe_submit_navdp_plan(
            observation,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )
        self._consume_navdp_plan(frame_id=int(observation.frame_id))
        self._refresh_stale_navdp_hold()
        plan_stop = bool(self._state.trajectory.planner_control_mode == "stop")
        return self._build_update(
            frame_id=observation.frame_id,
            action_command=action_command,
            stop=plan_stop,
            sensor_meta=observation.sensor_meta,
        )

    def _maybe_update_system2(
        self,
        observation: ExecutionObservation,
        *,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> None:
        planner = getattr(self._transport, "system2_planner", None)
        if planner is not None:
            self._consume_async_system2_result(
                observation,
                robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(robot_yaw),
            )
            if not self._system2_query_due(now=time.monotonic()):
                return
            if planner.has_pending_work():
                return
            planner.submit(
                AsyncSystem2Input(
                    frame_id=int(observation.frame_id),
                    rgb=observation.rgb,
                    depth=observation.depth,
                    stamp_s=time.monotonic(),
                )
            )
            self._state.goal.system2_submit_ts = time.monotonic()
            return

        now = time.monotonic()
        if not self._system2_query_due(now=now):
            return
        if getattr(self._transport, "system2_client", None) is None:
            return
        try:
            result = self._transport.system2_client.step_session(
                rgb=observation.rgb,
                depth=observation.depth,
                stamp_s=now,
            )
        except Exception as exc:  # noqa: BLE001
            self._state.trajectory.stats = PlannerStats(
                successful_calls=int(self._state.trajectory.stats.successful_calls),
                failed_calls=int(self._state.trajectory.stats.failed_calls) + 1,
                latency_ms=0.0,
                last_error=f"{type(exc).__name__}: {exc}",
                last_plan_step=int(observation.frame_id),
            )
            return
        self._state.goal.system2_submit_ts = now
        self._state.goal.system2_response_ts = time.monotonic()
        self._state.goal.system2_result_version += 1
        self._apply_system2_result(
            observation,
            result=result,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )

    def _system2_query_due(self, *, now: float) -> bool:
        interval = max(float(getattr(self._args, "s2_period_sec", 1.0)), 0.05)
        last_query_ts = max(float(self._state.goal.system2_submit_ts), float(self._state.goal.system2_response_ts))
        return self._state.goal.system2_result_version < 0 or (float(now) - last_query_ts) >= interval

    def _consume_async_system2_result(
        self,
        observation: ExecutionObservation,
        *,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> None:
        planner = getattr(self._transport, "system2_planner", None)
        if planner is None:
            return
        latest = planner.consume_latest(self._state.goal.system2_result_version)
        if latest is None:
            return
        self._state.goal.system2_result_version = int(latest.result_version)
        self._state.goal.system2_response_ts = time.monotonic()
        self._apply_system2_result(
            observation,
            result=latest.result,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )

    def _apply_system2_result(
        self,
        observation: ExecutionObservation,
        *,
        result: System2Result,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> None:
        decision_mode = str(result.decision_mode or "wait").strip().lower() or "wait"
        self._state.goal.system2_pixel_goal = None
        if result.pixel_xy is not None:
            pixel_xy = np.asarray(result.pixel_xy, dtype=np.float32).reshape(2)
            self._state.goal.system2_pixel_goal = [
                int(round(float(pixel_xy[0]))),
                int(round(float(pixel_xy[1]))),
            ]
        if decision_mode == "pixel_goal" and result.pixel_xy is not None:
            pixel_xy = tuple(int(round(float(v))) for v in np.asarray(result.pixel_xy, dtype=np.float32).reshape(2))
            world_xy = resolve_goal_world_xy_from_pixel(
                pixel_xy=pixel_xy,
                depth_image=observation.depth,
                intrinsic=observation.intrinsic,
                camera_pos_world=observation.cam_pos,
                camera_quat_wxyz=observation.cam_quat,
                window_size=int(getattr(self._args, "internvla_goal_depth_window", 11)),
                depth_min_m=float(getattr(self._args, "internvla_goal_depth_min", 0.1)),
                depth_max_m=float(getattr(self._args, "internvla_goal_depth_max", 6.0)),
            )
            if world_xy is not None:
                self._state.goal.goal_version += 1
                self._state.goal.local_xy = world_goal_to_robot_frame(
                    goal_xy=np.asarray(world_xy, dtype=np.float32),
                    robot_xy=np.asarray(robot_pos_world, dtype=np.float32),
                    robot_yaw=float(robot_yaw),
                ).astype(np.float32)
                self._state.trajectory.planner_control_mode = "trajectory"
                self._state.trajectory.planner_yaw_delta_rad = None
                self._state.trajectory.planner_control_reason = "pixel_goal"
            return
        if decision_mode in {"forward", "yaw_left", "yaw_right", "stop", "wait"}:
            self._state.trajectory.planner_control_version += 1
            if decision_mode in {"forward", "yaw_left", "yaw_right", "stop"}:
                self._state.goal.local_xy = np.zeros(2, dtype=np.float32)
                self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
            self._state.trajectory.planner_control_mode = decision_mode
            self._state.trajectory.planner_yaw_delta_rad = None
            self._state.trajectory.planner_control_reason = str(result.text)
            if decision_mode in {"stop", "wait"}:
                self._state.trajectory.used_cached_traj = False
            return
        if decision_mode == "look_down":
            if self._state.trajectory.trajectory_world.shape[0] == 0:
                self._state.trajectory.planner_control_version += 1
                self._state.trajectory.planner_control_mode = "wait"
                self._state.trajectory.planner_control_reason = str(result.text)

    def _maybe_submit_navdp_plan(
        self,
        observation: ExecutionObservation,
        *,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> None:
        if self._state.trajectory.planner_control_mode not in {None, "trajectory"}:
            return
        if not np.any(np.isfinite(self._state.goal.local_xy)):
            return
        if not np.any(np.abs(self._state.goal.local_xy) > 0.0):
            return
        now = time.monotonic()
        should_plan = (
            self._state.trajectory.trajectory_world.shape[0] == 0
            or (now - float(self._state.trajectory.last_plan_stamp_s)) >= max(float(getattr(self._args, "s1_period_sec", 0.2)), 0.05)
        )
        if not should_plan:
            return
        self._transport.pointgoal_planner.submit(
            PlannerInput(
                frame_id=int(observation.frame_id),
                local_goal_xy=np.asarray(self._state.goal.local_xy, dtype=np.float32),
                rgb=observation.rgb,
                depth=observation.depth,
                sensor_meta=self._nav_sensor_meta(
                    observation.sensor_meta,
                    robot_pos_world=robot_pos_world,
                    robot_yaw=robot_yaw,
                ),
                cam_pos=observation.cam_pos,
                cam_quat=observation.cam_quat,
                robot_pos=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(robot_yaw),
            )
        )

    def _consume_navdp_plan(self, *, frame_id: int) -> None:
        latest = self._transport.pointgoal_planner.consume_latest(self._state.trajectory.plan_version)
        success, failed, error, latency_ms = self._transport.pointgoal_planner.snapshot_status()
        if latest is not None:
            self._state.trajectory.plan_version = int(latest.plan_version)
            self._state.trajectory.trajectory_world = np.asarray(latest.trajectory_world, dtype=np.float32).copy()
            self._state.trajectory.last_plan_stamp_s = time.monotonic()
            self._state.goal.traj_version = int(latest.plan_version)
            self._state.trajectory.planner_control_mode = "trajectory"
            self._state.trajectory.planner_control_reason = ""
            self._state.trajectory.used_cached_traj = False
            self._state.trajectory.stale_sec = 0.0
            last_plan_step = int(latest.source_frame_id)
        else:
            last_plan_step = int(frame_id)
        self._state.trajectory.stats = PlannerStats(
            successful_calls=int(success),
            failed_calls=int(failed),
            latency_ms=float(latency_ms),
            last_error=str(error),
            last_plan_step=last_plan_step,
        )

    def _refresh_stale_navdp_hold(self) -> None:
        if self._state.trajectory.planner_control_mode not in {None, "trajectory"}:
            self._state.trajectory.stale_sec = -1.0
            return
        if self._state.trajectory.trajectory_world.shape[0] == 0 or float(self._state.trajectory.last_plan_stamp_s) <= 0.0:
            self._state.trajectory.used_cached_traj = False
            self._state.trajectory.stale_sec = -1.0
            return
        stale_sec = max(0.0, time.monotonic() - float(self._state.trajectory.last_plan_stamp_s))
        self._state.trajectory.stale_sec = float(stale_sec)
        max_stale = float(getattr(self._args, "traj_max_stale_sec", 4.0))
        if stale_sec > max_stale:
            self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
            self._state.trajectory.used_cached_traj = False
            self._state.trajectory.planner_control_mode = "wait"
            self._state.trajectory.planner_control_reason = "stale_hold_timeout"
            return
        self._state.trajectory.used_cached_traj = bool(stale_sec > 0.0)

    def _run_interactive(
        self,
        observation: ExecutionObservation,
        *,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> TrajectoryUpdate:
        self._consume_interactive_controls()
        plan_stop = False
        if self._state.interactive.phase == "roaming":
            self._update_interactive_roaming(observation)
        else:
            plan_stop = self._update_interactive_task(
                observation,
                robot_pos_world=robot_pos_world,
                robot_yaw=robot_yaw,
            )
        return self._build_update(
            frame_id=observation.frame_id,
            action_command=action_command,
            stop=plan_stop,
            sensor_meta=observation.sensor_meta,
        )

    def _consume_interactive_controls(self) -> None:
        if not self._interactive_enabled():
            return
        interactive = self._state.interactive
        with interactive.lock:
            cancel_requested = bool(interactive.cancel_requested)
            pending_command_id = int(interactive.pending_command_id)
            pending_instruction = str(interactive.pending_instruction)
            interactive.cancel_requested = False
            interactive.pending_command_id = -1
            interactive.pending_instruction = ""

        if cancel_requested:
            self._activate_roaming("user cancel")
        if pending_command_id >= 0 and pending_instruction != "":
            self._activate_task(pending_command_id, pending_instruction)

    def _activate_roaming(self, reason: str) -> bool:
        if getattr(self._transport, "navdp_client", None) is None or getattr(self._transport, "nogoal_planner", None) is None:
            return False
        try:
            self._transport.navdp_client.navigator_reset(self._transport._intrinsic.copy(), batch_size=1)
        except Exception as exc:  # noqa: BLE001
            error = f"roaming navigator_reset failed: {type(exc).__name__}: {exc}"
            self._state.interactive.phase = "roaming"
            self._state.trajectory.stale_sec = -1.0
            self._state.goal.goal_version = -1
            self._state.goal.traj_version = -1
            self._state.goal.system2_pixel_goal = None
            self._state.trajectory.stats = PlannerStats(
                successful_calls=0,
                failed_calls=1,
                latency_ms=0.0,
                last_error=error,
                last_plan_step=int(self._state.trajectory.stats.last_plan_step),
            )
            self._interactive_clear_active_task()
            self._state.reset_planner_control()
            self._emit_interactive_trajectory(np.zeros((0, 3), dtype=np.float32))
            return False

        self._transport.nogoal_planner.reset_state()
        self._state.interactive.last_nogoal_plan_version = -1
        self._state.interactive.last_nogoal_failed_calls = 0
        self._state.interactive.last_nav_plan_version = -1
        self._state.interactive.last_nav_failed_calls = 0
        self._interactive_clear_active_task()
        self._state.reset_planner_control()
        self._state.interactive.phase = "roaming"
        self._state.trajectory.stale_sec = -1.0
        self._state.goal.goal_version = -1
        self._state.goal.traj_version = -1
        self._state.goal.system2_result_version = -1
        self._state.goal.system2_pixel_goal = None
        self._state.goal.system2_submit_ts = 0.0
        self._state.goal.system2_response_ts = time.perf_counter()
        self._state.trajectory.stats = PlannerStats()
        if getattr(self._transport, "system2_planner", None) is not None:
            self._transport.system2_planner.reset_state()
        self._emit_interactive_trajectory(np.zeros((0, 3), dtype=np.float32))
        return True

    def _activate_task(self, command_id: int, instruction: str) -> bool:
        if getattr(self._transport, "system2_client", None) is None:
            return False
        self._state.interactive.phase = "task_pending"
        self._emit_interactive_trajectory(np.zeros((0, 3), dtype=np.float32))
        if getattr(self._transport, "nogoal_planner", None) is not None:
            self._transport.nogoal_planner.reset_state()
        self._state.interactive.last_nogoal_plan_version = -1
        self._state.interactive.last_nogoal_failed_calls = 0
        self._state.interactive.last_nav_plan_version = -1
        self._state.interactive.last_nav_failed_calls = 0
        self._state.reset_planner_control()
        try:
            self.start_nav_task(instruction)
        except Exception as exc:  # noqa: BLE001
            error = f"start_nav_task failed for command_id={int(command_id)}: {type(exc).__name__}: {exc}"
            self._state.trajectory.stats = PlannerStats(
                successful_calls=0,
                failed_calls=1,
                latency_ms=0.0,
                last_error=error,
                last_plan_step=int(self._state.trajectory.stats.last_plan_step),
            )
            self._activate_roaming(f"nav task start failure for command_id={int(command_id)}")
            return False

        self._state.interactive.phase = "task_active"
        self._state.interactive.active_command_id = int(command_id)
        self._state.interactive.active_instruction = str(instruction)
        self._state.goal.nav_instruction = str(instruction)
        self._state.trajectory.stale_sec = -1.0
        self._state.goal.goal_version = -1
        self._state.goal.traj_version = -1
        self._state.goal.system2_result_version = -1
        self._state.goal.system2_pixel_goal = None
        self._state.goal.system2_submit_ts = 0.0
        self._state.goal.system2_response_ts = time.perf_counter()
        self._state.trajectory.stats = PlannerStats()
        return True

    def _update_interactive_roaming(self, observation: ExecutionObservation) -> None:
        should_plan = self._state.trajectory.trajectory_world.shape[0] == 0 or (
            observation.frame_id % max(int(getattr(self._args, "plan_interval_frames", 1)), 1) == 0
        )
        if should_plan:
            self._transport.nogoal_planner.submit(
                NoGoalPlannerInput(
                    frame_id=int(observation.frame_id),
                    rgb=observation.rgb,
                    depth=observation.depth,
                    sensor_meta=dict(observation.sensor_meta),
                    cam_pos=observation.cam_pos,
                    cam_quat=observation.cam_quat,
                )
            )

        plan = self._transport.nogoal_planner.consume_latest(last_seen_version=self._state.interactive.last_nogoal_plan_version)
        if plan is not None:
            self._state.interactive.last_nogoal_plan_version = int(plan.plan_version)
            self._emit_interactive_trajectory(plan.trajectory_world)
            self._state.trajectory.stats = PlannerStats(
                successful_calls=int(plan.successful_calls),
                failed_calls=int(plan.failed_calls),
                latency_ms=float(plan.latency_ms),
                last_error=str(plan.last_error),
                last_plan_step=int(plan.source_frame_id),
            )

        success_calls, failed_calls, last_error, planner_latency_ms = self._transport.nogoal_planner.snapshot_status()
        self._state.interactive.last_nogoal_failed_calls = int(failed_calls)
        self._state.trajectory.stats = PlannerStats(
            successful_calls=int(success_calls),
            failed_calls=int(failed_calls),
            latency_ms=float(planner_latency_ms),
            last_error=str(last_error),
            last_plan_step=int(self._state.trajectory.stats.last_plan_step),
        )

    def _update_interactive_task(
        self,
        observation: ExecutionObservation,
        *,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> bool:
        self._maybe_update_system2(
            observation,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )
        self._maybe_submit_navdp_plan(
            observation,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )
        self._consume_navdp_plan(frame_id=int(observation.frame_id))
        self._refresh_stale_navdp_hold()
        if self._state.trajectory.planner_control_mode == "stop":
            self._activate_roaming(f"task complete command_id={int(self._state.interactive.active_command_id)}")
            return True
        return False

    @staticmethod
    def _nav_sensor_meta(
        sensor_meta: dict[str, Any] | None,
        *,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, Any]:
        enriched = dict(sensor_meta) if isinstance(sensor_meta, dict) else {}
        robot_pose = np.asarray(robot_pos_world, dtype=np.float32).reshape(-1)
        if robot_pose.shape[0] >= 3 and np.all(np.isfinite(robot_pose[:3])):
            enriched["robot_pose_xyz"] = [float(v) for v in robot_pose[:3]]
        if np.isfinite(float(robot_yaw)):
            enriched["robot_yaw_rad"] = float(robot_yaw)
        return enriched

    def _consume_planner_latest(
        self,
        *,
        planner,
        frame_id: int,
        action_command: ActionCommand | None,
        goal_local_xy: np.ndarray | None = None,
        sensor_meta: dict[str, Any] | None = None,
    ) -> TrajectoryUpdate:
        deadline = time.time() + float(getattr(self._args, "plan_wait_timeout_sec", 0.5))
        latest: PlannerOutput | None = None
        while time.time() < deadline:
            latest = planner.consume_latest(self._state.trajectory.plan_version)
            if latest is not None:
                break
            time.sleep(0.01)
        success, failed, error, latency_ms = planner.snapshot_status()
        if latest is not None:
            self._state.trajectory.plan_version = int(latest.plan_version)
            self._state.trajectory.trajectory_world = np.asarray(latest.trajectory_world, dtype=np.float32).copy()
            source_frame_id = int(latest.source_frame_id)
        else:
            source_frame_id = int(frame_id)
        self._state.reset_planner_control()
        self._state.trajectory.stats = PlannerStats(
            successful_calls=int(success),
            failed_calls=int(failed),
            latency_ms=float(latency_ms),
            last_error=str(error),
            last_plan_step=source_frame_id,
        )
        return TrajectoryUpdate(
            trajectory_world=np.asarray(self._state.trajectory.trajectory_world, dtype=np.float32).copy(),
            plan_version=self._state.trajectory.plan_version,
            stats=self._state.trajectory.stats,
            source_frame_id=source_frame_id,
            goal_local_xy=goal_local_xy,
            action_command=action_command,
            stop=bool(action_command is None and self._state.trajectory.trajectory_world.shape[0] == 0),
            sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
        )

    def _emit_interactive_trajectory(self, trajectory_world: np.ndarray) -> None:
        traj = np.asarray(trajectory_world, dtype=np.float32)
        if traj.size == 0:
            traj = np.zeros((0, 3), dtype=np.float32)
        self._state.interactive.session_plan_version += 1
        self._state.trajectory.plan_version = int(self._state.interactive.session_plan_version)
        self._state.trajectory.last_plan_stamp_s = time.monotonic()
        self._state.trajectory.trajectory_world = traj.copy()

    def _interactive_clear_active_task(self) -> None:
        self._state.interactive.active_command_id = -1
        self._state.interactive.active_instruction = ""
        if self._interactive_enabled():
            self._state.goal.nav_instruction = ""

    def _build_update(
        self,
        *,
        frame_id: int,
        action_command: ActionCommand | None,
        stop: bool = False,
        sensor_meta: dict[str, Any] | None = None,
    ) -> TrajectoryUpdate:
        interactive_phase = None
        interactive_command_id = -1
        interactive_instruction = ""
        if self._interactive_enabled():
            interactive_phase = str(self._state.interactive.phase)
            interactive_command_id = int(self._state.interactive.active_command_id)
            interactive_instruction = str(self._state.interactive.active_instruction)
        return TrajectoryUpdate(
            trajectory_world=np.asarray(self._state.trajectory.trajectory_world, dtype=np.float32).copy(),
            plan_version=int(self._state.trajectory.plan_version),
            stats=self._state.trajectory.stats,
            source_frame_id=int(frame_id),
            goal_local_xy=np.asarray(self._state.goal.local_xy, dtype=np.float32).copy() if self._state.mode == "MEM_NAV" else None,
            action_command=action_command,
            stop=bool(stop),
            planner_control_mode=self._state.trajectory.planner_control_mode,
            planner_control_version=int(self._state.trajectory.planner_control_version),
            planner_yaw_delta_rad=self._state.trajectory.planner_yaw_delta_rad,
            stale_sec=float(self._state.trajectory.stale_sec),
            goal_version=int(self._state.goal.goal_version),
            traj_version=int(self._state.goal.traj_version),
            used_cached_traj=bool(self._state.trajectory.used_cached_traj),
            sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
            interactive_phase=interactive_phase,
            interactive_command_id=interactive_command_id,
            interactive_instruction=interactive_instruction,
        )

    def reset_for_mode(self, mode: str, *, reason: str = "") -> None:
        del reason
        if normalize_execution_mode(mode) == self._state.mode:
            return
        self._state.set_mode(mode)
        self._state.reset_navigation_state()
