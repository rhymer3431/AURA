from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from common.geometry import within_xy_radius, world_goal_to_robot_frame, wrap_to_pi
from control.async_planners import (
    NoGoalPlannerInput,
    PlannerInput,
    PlannerOutput,
)
from inference.vlm import AsyncSystem2Input, System2Result, normalized_uv_to_pixel_xy, resolve_goal_world_xy_from_pixel
from systems.transport.messages import ActionCommand
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
        self._transport.system2_client.reset(
            text,
            language=str(getattr(self._args, "nav_instruction_language", "auto")).strip() or "auto",
        )
        try:
            self._transport.navdp_client.navigator_reset(self._transport._intrinsic.copy(), batch_size=1)
        except Exception:
            pass
        self._state.reset_navigation_state()
        self._state.goal.nav_instruction = text
        self._state.interactive.last_nogoal_plan_version = -1
        self._state.interactive.last_nav_plan_version = -1
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
        self._advance_action_override(
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
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
        if self._state.action_override.mode is not None:
            return
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
            self._state.system2.last_error = f"{type(exc).__name__}: {exc}"
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
        interval = max(float(getattr(self._args, "s2_period_sec", 1.0)), 0.0)
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

    @staticmethod
    def _direct_action_modes() -> tuple[str, ...]:
        return ("forward", "yaw_left", "yaw_right")

    @staticmethod
    def _result_status(result: System2Result) -> str:
        return str(getattr(result, "status", "")).strip().lower()

    @staticmethod
    def _result_text(result: System2Result) -> str:
        return str(getattr(result, "text", "")).strip()

    @staticmethod
    def _result_decision_mode(result: System2Result) -> str:
        return str(getattr(result, "decision_mode", "wait")).strip().lower() or "wait"

    @staticmethod
    def _result_needs_requery(result: System2Result) -> bool:
        return bool(getattr(result, "needs_requery", False))

    @staticmethod
    def _result_action_sequence(result: System2Result) -> tuple[str, ...]:
        raw_sequence = getattr(result, "action_sequence", ()) or ()
        return tuple(str(item).strip().lower() for item in raw_sequence if str(item).strip() != "")

    @staticmethod
    def _result_stamp_s(result: System2Result) -> float:
        raw_stamp = getattr(result, "stamp_s", time.monotonic())
        try:
            return float(raw_stamp)
        except (TypeError, ValueError):
            return time.monotonic()

    def _system2_result_signature(self, observation: ExecutionObservation, result: System2Result) -> str:
        pixel_xy = self._normalize_result_pixel_xy(observation, result)
        pixel_token = "" if pixel_xy is None else f"{int(pixel_xy[0])},{int(pixel_xy[1])}"
        action_sequence = self._result_action_sequence(result)
        return "|".join(
            (
                self._result_status(result),
                self._result_decision_mode(result),
                self._result_text(result),
                pixel_token,
                ",".join(action_sequence),
                "1" if self._result_needs_requery(result) else "0",
            )
        )

    def _normalize_result_pixel_xy(self, observation: ExecutionObservation, result: System2Result) -> np.ndarray | None:
        if result.pixel_xy is not None:
            return np.asarray(result.pixel_xy, dtype=np.float32).reshape(2)
        if result.uv_norm is None:
            return None
        pixel_xy = normalized_uv_to_pixel_xy(
            np.asarray(result.uv_norm, dtype=np.float32).reshape(2),
            image_width=int(observation.rgb.shape[1]),
            image_height=int(observation.rgb.shape[0]),
        )
        return np.asarray(pixel_xy, dtype=np.float32)

    def _record_system2_result(self, observation: ExecutionObservation, result: System2Result, *, error: str = "") -> str:
        pixel_xy = self._normalize_result_pixel_xy(observation, result)
        signature = self._system2_result_signature(observation, result)
        action_sequence = self._result_action_sequence(result)
        payload: dict[str, object] = {
            "status": self._result_status(result),
            "text": self._result_text(result),
            "decision_mode": self._result_decision_mode(result),
            "needs_requery": self._result_needs_requery(result),
            "latency_ms": float(getattr(result, "latency_ms", 0.0) or 0.0),
            "stamp_s": self._result_stamp_s(result),
            "action_sequence": list(action_sequence),
        }
        if pixel_xy is not None:
            payload["pixel_xy"] = [int(round(float(pixel_xy[0]))), int(round(float(pixel_xy[1])))]
        self._state.system2.last_signature = signature
        self._state.system2.last_result = payload
        self._state.system2.last_error = str(error)
        self._state.system2.latest_decision_mode = str(payload["decision_mode"])
        self._state.goal.system2_pixel_goal = payload.get("pixel_xy") if isinstance(payload.get("pixel_xy"), list) else None
        return signature

    def _begin_goal_candidate(self, *, world_xy: np.ndarray, pixel_xy: np.ndarray, stamp_s: float) -> None:
        goal = self._state.goal
        world = np.asarray(world_xy, dtype=np.float32).reshape(2)
        pixel = np.asarray(pixel_xy, dtype=np.float32).reshape(2)
        goal.candidate_kind = "point"
        goal.raw_candidate_world_xy = world.copy()
        goal.filtered_candidate_world_xy = world.copy()
        goal.raw_candidate_pixel_xy = pixel.copy()
        goal.filtered_candidate_pixel_xy = pixel.copy()
        goal.candidate_started_at_s = float(stamp_s)
        goal.candidate_last_stamp_s = float(stamp_s)
        goal.candidate_sample_count = 1

    def _update_goal_candidate(self, *, world_xy: np.ndarray, pixel_xy: np.ndarray, stamp_s: float) -> None:
        goal = self._state.goal
        world = np.asarray(world_xy, dtype=np.float32).reshape(2)
        pixel = np.asarray(pixel_xy, dtype=np.float32).reshape(2)
        if goal.candidate_kind != "point" or goal.filtered_candidate_world_xy is None or goal.filtered_candidate_pixel_xy is None:
            self._begin_goal_candidate(world_xy=world, pixel_xy=pixel, stamp_s=stamp_s)
            return
        alpha = float(np.clip(float(getattr(self._args, "internvla_goal_filter_alpha", 0.35)), 0.0, 1.0))
        goal.raw_candidate_world_xy = world.copy()
        goal.filtered_candidate_world_xy = ((1.0 - alpha) * goal.filtered_candidate_world_xy) + (alpha * world)
        goal.raw_candidate_pixel_xy = pixel.copy()
        goal.filtered_candidate_pixel_xy = ((1.0 - alpha) * goal.filtered_candidate_pixel_xy) + (alpha * pixel)
        goal.candidate_last_stamp_s = float(stamp_s)
        goal.candidate_sample_count += 1

    def _goal_candidate_ready(self) -> bool:
        goal = self._state.goal
        if goal.candidate_kind != "point" or goal.filtered_candidate_world_xy is None:
            return False
        stable_time = max(0.0, float(goal.candidate_last_stamp_s - goal.candidate_started_at_s))
        return (
            int(goal.candidate_sample_count) >= max(int(getattr(self._args, "internvla_goal_confirm_samples", 2)), 1)
            or stable_time >= max(float(getattr(self._args, "internvla_goal_min_stable_time", 0.6)), 0.0)
        )

    def _clear_goal(self, reason: str) -> None:
        had_goal = self._state.has_active_goal() or self._state.has_pending_goal()
        self._state.goal.target_world_xy = None
        self._state.goal.target_pixel_xy = None
        self._state.goal.local_xy = np.zeros(2, dtype=np.float32)
        self._state.goal.last_clear_reason = str(reason)
        self._state.clear_goal_candidate()
        if had_goal:
            self._state.goal.goal_version += 1
        self._state.goal.traj_version = -1

    def _commit_world_goal(
        self,
        *,
        world_xy: np.ndarray,
        pixel_xy: np.ndarray,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> None:
        world = np.asarray(world_xy, dtype=np.float32).reshape(2)
        pixel = np.asarray(np.rint(pixel_xy), dtype=np.float32).reshape(2)
        self._state.goal.target_world_xy = world.copy()
        self._state.goal.target_pixel_xy = pixel.copy()
        self._state.goal.local_xy = world_goal_to_robot_frame(
            goal_xy=world,
            robot_xy=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
        ).astype(np.float32)
        self._state.goal.goal_version += 1
        self._state.goal.last_clear_reason = ""
        self._state.clear_goal_candidate()
        self._state.navdp.last_discard_reason = ""
        self._state.trajectory.planner_control_mode = "trajectory"
        self._state.trajectory.planner_control_reason = "pixel_goal"
        self._state.trajectory.planner_control_queue = ()
        self._state.trajectory.planner_control_progress = 0.0
        self._state.trajectory.planner_yaw_delta_rad = None
        self._state.trajectory.stale_hold_reason = ""

    def _set_direct_action_override(
        self,
        *,
        modes: tuple[str, ...],
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        reason: str,
    ) -> None:
        active_mode = str(modes[0])
        pending_modes = tuple(str(item) for item in modes[1:])
        self._state.action_override.mode = active_mode
        self._state.action_override.pending_modes = pending_modes
        self._state.action_override.started_at_s = time.monotonic()
        self._state.action_override.start_pos_xy = np.asarray(robot_pos_world, dtype=np.float32).reshape(-1)[:2].copy()
        self._state.action_override.start_yaw_rad = float(robot_yaw)
        self._state.action_override.progress = 0.0
        self._state.action_override.target_distance_m = (
            max(0.05, float(getattr(self._args, "internvla_forward_step_m", 0.5)))
            if active_mode == "forward"
            else 0.0
        )
        self._state.action_override.target_yaw_rad = (
            float(np.deg2rad(max(1.0, float(getattr(self._args, "internvla_turn_step_deg", 30.0)))))
            if active_mode in {"yaw_left", "yaw_right"}
            else 0.0
        )
        self._state.trajectory.planner_control_version += 1
        self._state.trajectory.planner_control_mode = active_mode
        self._state.trajectory.planner_control_queue = pending_modes
        self._state.trajectory.planner_control_progress = 0.0
        self._state.trajectory.planner_control_reason = str(reason)
        self._state.trajectory.planner_yaw_delta_rad = None
        self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
        self._state.trajectory.used_cached_traj = False
        self._state.trajectory.stale_hold_reason = ""
        self._state.locomotion.state_label = "waiting"
        self._state.locomotion.last_command = np.zeros(3, dtype=np.float32)
        self._state.locomotion.last_command_stamp = time.monotonic()

    def _clear_action_override(self) -> None:
        self._state.action_override.mode = None
        self._state.action_override.pending_modes = ()
        self._state.action_override.started_at_s = 0.0
        self._state.action_override.start_pos_xy = None
        self._state.action_override.start_yaw_rad = 0.0
        self._state.action_override.target_distance_m = 0.0
        self._state.action_override.target_yaw_rad = 0.0
        self._state.action_override.progress = 0.0

    def _advance_action_override(self, *, robot_pos_world: np.ndarray, robot_yaw: float) -> None:
        mode = self._state.action_override.mode
        if mode is None:
            self._state.trajectory.planner_control_queue = ()
            self._state.trajectory.planner_control_progress = 0.0
            return
        now = time.monotonic()
        progress = 0.0
        completed = False
        if mode == "forward":
            start_xy = self._state.action_override.start_pos_xy
            target_distance = max(float(self._state.action_override.target_distance_m), 1.0e-6)
            if start_xy is not None:
                distance = float(np.linalg.norm(np.asarray(robot_pos_world, dtype=np.float32)[:2] - np.asarray(start_xy, dtype=np.float32)))
                progress = min(distance, target_distance) / target_distance
                completed = distance >= (target_distance - 1.0e-4)
        elif mode in {"yaw_left", "yaw_right"}:
            target_yaw = max(float(self._state.action_override.target_yaw_rad), 1.0e-6)
            yaw_delta = abs(float(wrap_to_pi(float(robot_yaw) - float(self._state.action_override.start_yaw_rad))))
            progress = min(yaw_delta, target_yaw) / target_yaw
            completed = yaw_delta >= (target_yaw - 1.0e-4)
        elapsed = max(0.0, now - float(self._state.action_override.started_at_s))
        timeout_s = max(0.1, float(getattr(self._args, "internvla_action_timeout_s", 3.0)))
        self._state.action_override.progress = float(progress)
        self._state.trajectory.planner_control_progress = float(progress)
        self._state.trajectory.planner_control_queue = tuple(self._state.action_override.pending_modes)
        if not completed and progress < 1.0 and elapsed <= timeout_s:
            return
        pending_modes = tuple(self._state.action_override.pending_modes)
        if pending_modes:
            self._set_direct_action_override(
                modes=pending_modes,
                robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(robot_yaw),
                reason="direct_action_queue",
            )
            return
        self._clear_action_override()
        self._state.trajectory.planner_control_version += 1
        self._state.trajectory.planner_control_mode = "wait"
        self._state.trajectory.planner_control_queue = ()
        self._state.trajectory.planner_control_progress = 1.0
        self._state.trajectory.planner_control_reason = "direct_action_complete"
        self._state.trajectory.planner_yaw_delta_rad = None
        self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
        self._state.trajectory.used_cached_traj = False
        self._state.trajectory.stale_hold_reason = ""

    def _apply_system2_result(
        self,
        observation: ExecutionObservation,
        *,
        result: System2Result,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> None:
        previous_signature = str(self._state.system2.last_signature)
        decision_mode = self._result_decision_mode(result)
        result_signature = self._record_system2_result(observation, result)
        same_signature = result_signature == previous_signature
        pixel_xy = self._normalize_result_pixel_xy(observation, result)
        if decision_mode == "pixel_goal":
            self._clear_action_override()
            if pixel_xy is None:
                self._state.system2.last_error = "InternVLA pixel goal is missing both pixel_xy and uv_norm."
                return
            world_xy = resolve_goal_world_xy_from_pixel(
                pixel_xy=(int(round(float(pixel_xy[0]))), int(round(float(pixel_xy[1])))),
                depth_image=observation.depth,
                intrinsic=observation.intrinsic,
                camera_pos_world=observation.cam_pos,
                camera_quat_wxyz=observation.cam_quat,
                window_size=int(getattr(self._args, "internvla_goal_depth_window", 5)),
                depth_min_m=float(getattr(self._args, "internvla_goal_depth_min", 0.25)),
                depth_max_m=float(getattr(self._args, "internvla_goal_depth_max", 6.0)),
            )
            if world_xy is None:
                self._state.system2.last_error = (
                    "InternVLA goal projection failed: "
                    f"no valid depth near pixel={[int(round(float(pixel_xy[0]))), int(round(float(pixel_xy[1])))]}"
                )
                return
            current_goal_world = self._state.goal.target_world_xy
            min_goal_dist = max(float(getattr(self._args, "internvla_goal_update_min_dist", 0.35)), 0.0)
            if current_goal_world is not None:
                active_distance = float(
                    np.linalg.norm(np.asarray(world_xy, dtype=np.float32).reshape(2) - np.asarray(current_goal_world, dtype=np.float32).reshape(2))
                )
                if active_distance < min_goal_dist:
                    self._state.clear_goal_candidate()
                    self._state.trajectory.planner_control_mode = "trajectory"
                self._state.trajectory.planner_control_reason = "pixel_goal_unchanged"
                return
            candidate_world = self._state.goal.filtered_candidate_world_xy
            if candidate_world is None:
                self._begin_goal_candidate(
                    world_xy=np.asarray(world_xy, dtype=np.float32),
                    pixel_xy=pixel_xy,
                    stamp_s=self._result_stamp_s(result),
                )
            else:
                candidate_distance = float(
                    np.linalg.norm(np.asarray(world_xy, dtype=np.float32).reshape(2) - np.asarray(candidate_world, dtype=np.float32).reshape(2))
                )
                if candidate_distance >= min_goal_dist:
                    self._begin_goal_candidate(
                        world_xy=np.asarray(world_xy, dtype=np.float32),
                        pixel_xy=pixel_xy,
                        stamp_s=self._result_stamp_s(result),
                    )
                else:
                    self._update_goal_candidate(
                        world_xy=np.asarray(world_xy, dtype=np.float32),
                        pixel_xy=pixel_xy,
                        stamp_s=self._result_stamp_s(result),
                    )
            if not self._goal_candidate_ready():
                self._state.trajectory.planner_control_mode = "wait"
                self._state.trajectory.planner_control_reason = "pixel_goal_pending"
                self._state.trajectory.planner_control_queue = ()
                self._state.trajectory.planner_control_progress = 0.0
                return
            stabilized_world = self._state.goal.filtered_candidate_world_xy
            stabilized_pixel = self._state.goal.filtered_candidate_pixel_xy
            if stabilized_world is None or stabilized_pixel is None:
                return
            self._commit_world_goal(
                world_xy=np.asarray(stabilized_world, dtype=np.float32),
                pixel_xy=np.asarray(stabilized_pixel, dtype=np.float32),
                robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(robot_yaw),
            )
            return
        if decision_mode in self._direct_action_modes():
            action_sequence = tuple(mode for mode in (self._result_action_sequence(result) or (decision_mode,)) if mode in self._direct_action_modes())
            if not action_sequence:
                action_sequence = (decision_mode,)
            self._clear_goal(f"system2_{decision_mode}")
            self._state.navdp.request_active = False
            self._state.navdp.last_discard_reason = ""
            self._state.navdp.error = ""
            self._set_direct_action_override(
                modes=action_sequence,
                robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(robot_yaw),
                reason=self._result_text(result),
            )
            return
        if decision_mode == "stop" or self._result_status(result) == "stop":
            self._clear_action_override()
            self._clear_goal("stop")
            self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
            self._state.trajectory.used_cached_traj = False
            self._state.trajectory.stale_hold_reason = ""
            self._state.trajectory.planner_control_version += 1
            self._state.trajectory.planner_control_mode = "stop"
            self._state.trajectory.planner_control_queue = ()
            self._state.trajectory.planner_control_progress = 0.0
            self._state.trajectory.planner_control_reason = self._result_text(result)
            self._state.navdp.request_active = False
            self._state.navdp.last_discard_reason = ""
            return
        if decision_mode in {"wait", "look_down"}:
            self._clear_action_override()
            if self._state.has_active_goal():
                self._state.trajectory.planner_control_mode = "trajectory"
                self._state.trajectory.planner_control_reason = "wait_preserve_goal"
                self._state.trajectory.planner_control_queue = ()
                self._state.trajectory.planner_control_progress = 0.0
                if decision_mode == "look_down":
                    self._state.trajectory.stale_hold_reason = "look_down_requery"
            elif self._state.has_pending_goal():
                self._state.trajectory.planner_control_version += 1
                self._state.trajectory.planner_control_mode = "wait"
                self._state.trajectory.planner_control_reason = "wait_pending_goal"
                self._state.trajectory.planner_control_queue = ()
                self._state.trajectory.planner_control_progress = 0.0
                self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
                self._state.trajectory.used_cached_traj = False
                if decision_mode == "look_down":
                    self._state.trajectory.stale_hold_reason = "look_down_requery"
            else:
                self._clear_goal("wait")
                self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
                self._state.trajectory.used_cached_traj = False
                self._state.trajectory.stale_hold_reason = "look_down_requery" if decision_mode == "look_down" else ""
                self._state.trajectory.planner_control_version += 1
                self._state.trajectory.planner_control_mode = "wait"
                self._state.trajectory.planner_control_queue = ()
                self._state.trajectory.planner_control_progress = 0.0
                self._state.trajectory.planner_control_reason = self._result_text(result) or decision_mode
            return
        if self._result_status(result) == "error":
            self._state.system2.last_error = f"InternVLA server error: {self._result_text(result)}"
        if same_signature and self._state.system2.last_error == "":
            self._state.system2.latest_decision_mode = decision_mode

    def _maybe_submit_navdp_plan(
        self,
        observation: ExecutionObservation,
        *,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> None:
        if self._state.trajectory.planner_control_mode not in {None, "trajectory"}:
            return
        if self._state.action_override.mode is not None:
            return
        if self._state.goal.target_world_xy is None:
            self._state.navdp.request_active = False
            return
        goal_tolerance = max(float(getattr(self._args, "goal_tolerance_m", 0.4)), 0.0)
        if within_xy_radius(
            np.asarray([float(robot_pos_world[0]), float(robot_pos_world[1]), 0.0], dtype=np.float32),
            np.asarray([float(self._state.goal.target_world_xy[0]), float(self._state.goal.target_world_xy[1]), 0.0], dtype=np.float32),
            goal_tolerance,
        ):
            self._state.goal.local_xy = np.zeros(2, dtype=np.float32)
            self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
            self._state.trajectory.used_cached_traj = False
            self._state.trajectory.stale_hold_reason = ""
            self._state.trajectory.planner_control_mode = "wait"
            self._state.trajectory.planner_control_reason = "goal_reached"
            self._state.trajectory.planner_control_queue = ()
            self._state.trajectory.planner_control_progress = 0.0
            self._state.navdp.request_active = False
            return
        self._state.goal.local_xy = world_goal_to_robot_frame(
            goal_xy=np.asarray(self._state.goal.target_world_xy, dtype=np.float32),
            robot_xy=np.asarray(robot_pos_world, dtype=np.float32)[:2],
            robot_yaw=float(robot_yaw),
        ).astype(np.float32)
        if not np.any(np.isfinite(self._state.goal.local_xy)):
            return
        now = time.monotonic()
        replan_period = 1.0 / max(float(getattr(self._args, "navdp_replan_hz", 3.0)), 0.1)
        should_plan = (
            self._state.trajectory.trajectory_world.shape[0] == 0
            or int(self._state.navdp.last_request_goal_version) != int(self._state.goal.goal_version)
            or (now - float(self._state.navdp.last_request_started_at_s)) >= replan_period
        )
        if not should_plan:
            return
        self._state.navdp.request_active = True
        self._state.navdp.last_request_goal_version = int(self._state.goal.goal_version)
        self._state.navdp.last_request_mode = str(self._state.goal_mode())
        self._state.navdp.last_request_started_at_s = float(now)
        self._state.navdp.last_request_source_frame_id = int(observation.frame_id)
        self._state.navdp.error = ""
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
            current_goal_version = int(self._state.goal.goal_version)
            request_goal_version = int(self._state.navdp.last_request_goal_version)
            request_source_frame_id = int(self._state.navdp.last_request_source_frame_id)
            request_mode = str(self._state.navdp.last_request_mode)
            current_goal_mode = self._state.goal_mode()
            discard_reason = ""
            if request_mode not in {"point", "pixel"}:
                discard_reason = f"unsupported_goal_mode:{request_mode or 'none'}"
            elif current_goal_version != request_goal_version:
                discard_reason = f"goal_version_changed:{request_goal_version}->{current_goal_version}"
            elif current_goal_mode != request_mode:
                discard_reason = "active_goal_changed"
            elif request_source_frame_id >= 0 and int(latest.source_frame_id) != request_source_frame_id:
                discard_reason = f"request_frame_mismatch:{request_source_frame_id}->{int(latest.source_frame_id)}"

            if discard_reason == "":
                self._state.trajectory.plan_version = int(latest.plan_version)
                self._state.trajectory.trajectory_world = np.asarray(latest.trajectory_world, dtype=np.float32).copy()
                self._state.trajectory.last_plan_stamp_s = time.monotonic()
                self._state.goal.traj_version = int(latest.plan_version)
                self._state.trajectory.planner_control_mode = "trajectory"
                self._state.trajectory.planner_control_reason = ""
                self._state.trajectory.planner_control_queue = ()
                self._state.trajectory.planner_control_progress = 0.0
                self._state.trajectory.used_cached_traj = False
                self._state.trajectory.stale_sec = 0.0
                self._state.trajectory.stale_hold_reason = ""
                self._state.navdp.last_committed_goal_version = current_goal_version
                self._state.navdp.last_committed_plan_version = int(latest.plan_version)
                self._state.navdp.last_discard_reason = ""
                self._state.navdp.request_active = False
                self._state.navdp.error = ""
                last_plan_step = int(latest.source_frame_id)
            else:
                self._state.navdp.last_discarded_goal_version = request_goal_version
                self._state.navdp.last_discard_reason = discard_reason
                self._state.navdp.request_active = False
                last_plan_step = int(frame_id)
        else:
            last_plan_step = int(frame_id)
            if error != "":
                self._state.navdp.error = str(error)
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
        if self._state.goal.target_world_xy is None:
            self._state.trajectory.used_cached_traj = False
            self._state.trajectory.stale_sec = -1.0
            self._state.trajectory.stale_hold_reason = ""
            return
        if self._state.trajectory.trajectory_world.shape[0] == 0 or float(self._state.trajectory.last_plan_stamp_s) <= 0.0:
            self._state.trajectory.used_cached_traj = False
            self._state.trajectory.stale_sec = -1.0
            if self._state.trajectory.trajectory_world.shape[0] == 0 and str(self._state.trajectory.stale_hold_reason).strip() == "":
                self._state.trajectory.stale_hold_reason = "no_plan"
            return
        stale_sec = max(0.0, time.monotonic() - float(self._state.trajectory.last_plan_stamp_s))
        self._state.trajectory.stale_sec = float(stale_sec)
        plan_timeout = max(0.1, float(getattr(self._args, "navdp_plan_timeout", 1.5)))
        hold_timeout = max(plan_timeout, float(getattr(self._args, "navdp_hold_last_plan_timeout", 4.0)))
        if stale_sec > hold_timeout:
            self._state.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
            self._state.trajectory.used_cached_traj = False
            self._state.trajectory.stale_hold_reason = "hold_timeout"
            return
        if stale_sec > plan_timeout:
            self._state.trajectory.used_cached_traj = True
            self._state.trajectory.stale_hold_reason = "stale_hold"
            return
        self._state.trajectory.used_cached_traj = False
        self._state.trajectory.stale_hold_reason = ""

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
        self._advance_action_override(
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
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
            goal_local_xy=np.asarray(self._state.goal.local_xy, dtype=np.float32).copy()
            if self._state.mode in {"NAV", "MEM_NAV"}
            else None,
            action_command=action_command,
            stop=bool(stop),
            planner_control_mode=self._state.trajectory.planner_control_mode,
            planner_control_version=int(self._state.trajectory.planner_control_version),
            planner_control_reason=str(self._state.trajectory.planner_control_reason),
            planner_yaw_delta_rad=self._state.trajectory.planner_yaw_delta_rad,
            planner_control_queue=tuple(self._state.trajectory.planner_control_queue),
            planner_control_progress=float(self._state.trajectory.planner_control_progress),
            stale_sec=float(self._state.trajectory.stale_sec),
            stale_hold_reason=str(self._state.trajectory.stale_hold_reason),
            goal_version=int(self._state.goal.goal_version),
            traj_version=int(self._state.goal.traj_version),
            used_cached_traj=bool(self._state.trajectory.used_cached_traj),
            locomotion_state_label=str(self._state.locomotion.state_label),
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
