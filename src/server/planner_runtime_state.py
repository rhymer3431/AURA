from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from runtime.planning_session import PlannerStats
from schemas.execution_mode import ExecutionMode, normalize_execution_mode


def _zeros2() -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def _zeros3() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


def _empty_traj() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def _float_pair(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] < 2:
        return None
    return [float(array[0]), float(array[1])]


def _int_pair(value: np.ndarray | None) -> list[int] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] < 2:
        return None
    return [int(round(float(array[0]))), int(round(float(array[1])))]


@dataclass
class GoalState:
    local_xy: np.ndarray = field(default_factory=_zeros2)
    target_world_xy: np.ndarray | None = None
    target_pixel_xy: np.ndarray | None = None
    goal_version: int = -1
    traj_version: int = -1
    system2_result_version: int = -1
    system2_pixel_goal: list[int] | None = None
    system2_submit_ts: float = 0.0
    system2_response_ts: float = field(default_factory=time.perf_counter)
    nav_instruction: str = ""
    candidate_kind: str = "none"
    raw_candidate_world_xy: np.ndarray | None = None
    filtered_candidate_world_xy: np.ndarray | None = None
    raw_candidate_pixel_xy: np.ndarray | None = None
    filtered_candidate_pixel_xy: np.ndarray | None = None
    candidate_started_at_s: float = 0.0
    candidate_last_stamp_s: float = 0.0
    candidate_sample_count: int = 0
    last_clear_reason: str = ""


@dataclass
class TrajectoryState:
    trajectory_world: np.ndarray = field(default_factory=_empty_traj)
    plan_version: int = -1
    last_plan_stamp_s: float = 0.0
    planner_control_mode: str | None = None
    planner_control_version: int = -1
    planner_yaw_delta_rad: float | None = None
    planner_control_reason: str = ""
    planner_control_queue: tuple[str, ...] = ()
    planner_control_progress: float = 0.0
    used_cached_traj: bool = False
    stale_sec: float = -1.0
    stale_hold_reason: str = ""
    stats: PlannerStats = field(default_factory=PlannerStats)


@dataclass
class System2State:
    last_signature: str = ""
    last_result: dict[str, Any] = field(default_factory=dict)
    last_error: str = ""
    latest_decision_mode: str = ""


@dataclass
class NavDPState:
    last_request_goal_version: int = -1
    last_request_mode: str = ""
    last_request_started_at_s: float = 0.0
    last_request_source_frame_id: int = -1
    last_committed_goal_version: int = -1
    last_committed_plan_version: int = -1
    last_discarded_goal_version: int = -1
    last_discard_reason: str = ""
    request_active: bool = False
    error: str = ""


@dataclass
class ActionOverrideState:
    mode: str | None = None
    pending_modes: tuple[str, ...] = ()
    started_at_s: float = 0.0
    start_pos_xy: np.ndarray | None = None
    start_yaw_rad: float = 0.0
    target_distance_m: float = 0.0
    target_yaw_rad: float = 0.0
    progress: float = 0.0


@dataclass
class LocomotionState:
    state_label: str = ""
    last_command: np.ndarray = field(default_factory=_zeros3)
    last_command_stamp: float = 0.0


@dataclass
class InteractiveTaskState:
    phase: str = ""
    command_seq: int = 0
    pending_command_id: int = -1
    pending_instruction: str = ""
    cancel_requested: bool = False
    active_command_id: int = -1
    active_instruction: str = ""
    session_plan_version: int = -1
    last_nogoal_plan_version: int = -1
    last_nav_plan_version: int = -1
    last_nogoal_failed_calls: int = 0
    last_nav_failed_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class GlobalRouteState:
    planner: Any | None = None
    waypoints_world: list[tuple[float, float]] = field(default_factory=list)
    final_goal_xy: np.ndarray = field(default_factory=_zeros2)
    active_index: int = 0
    last_progress_ts: float = 0.0
    best_distance_m: float = float("inf")
    last_replan_reason: str = ""
    last_error: str = ""


@dataclass
class PlannerRuntimeState:
    mode: ExecutionMode
    goal: GoalState = field(default_factory=GoalState)
    trajectory: TrajectoryState = field(default_factory=TrajectoryState)
    system2: System2State = field(default_factory=System2State)
    navdp: NavDPState = field(default_factory=NavDPState)
    action_override: ActionOverrideState = field(default_factory=ActionOverrideState)
    locomotion: LocomotionState = field(default_factory=LocomotionState)
    interactive: InteractiveTaskState = field(default_factory=InteractiveTaskState)
    global_route: GlobalRouteState = field(default_factory=GlobalRouteState)
    launcher_processes: dict[str, subprocess.Popen[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = normalize_execution_mode(self.mode)

    def active_memory_instruction(self) -> str:
        if self.mode == "NAV" and str(self.interactive.active_instruction).strip() != "":
            return str(self.interactive.active_instruction).strip()
        if self.mode == "NAV":
            return str(self.goal.nav_instruction).strip()
        return ""

    def set_mode(self, mode: ExecutionMode) -> None:
        self.mode = normalize_execution_mode(mode)

    def goal_mode(self) -> str:
        if self.goal.target_world_xy is not None:
            return "point"
        if self.goal.target_pixel_xy is not None:
            return "pixel"
        return "none"

    def has_active_goal(self) -> bool:
        return self.goal_mode() in {"point", "pixel"}

    def has_pending_goal(self) -> bool:
        return self.goal.candidate_kind in {"point", "pixel"}

    def clear_goal_candidate(self) -> None:
        self.goal.candidate_kind = "none"
        self.goal.raw_candidate_world_xy = None
        self.goal.filtered_candidate_world_xy = None
        self.goal.raw_candidate_pixel_xy = None
        self.goal.filtered_candidate_pixel_xy = None
        self.goal.candidate_started_at_s = 0.0
        self.goal.candidate_last_stamp_s = 0.0
        self.goal.candidate_sample_count = 0

    def reset_navigation_state(self) -> None:
        self.goal.local_xy = _zeros2()
        self.goal.target_world_xy = None
        self.goal.target_pixel_xy = None
        self.goal.goal_version = -1
        self.goal.traj_version = -1
        self.goal.system2_result_version = -1
        self.goal.system2_pixel_goal = None
        self.goal.system2_submit_ts = 0.0
        self.goal.system2_response_ts = time.perf_counter()
        self.goal.nav_instruction = ""
        self.goal.last_clear_reason = ""
        self.clear_goal_candidate()

        self.trajectory.trajectory_world = _empty_traj()
        self.trajectory.plan_version = -1
        self.trajectory.last_plan_stamp_s = 0.0
        self.trajectory.stale_sec = -1.0
        self.trajectory.stale_hold_reason = ""
        self.trajectory.stats = PlannerStats()
        self.reset_planner_control()

        self.system2 = System2State()
        self.navdp = NavDPState()
        self.action_override = ActionOverrideState()
        self.locomotion = LocomotionState()

        self.clear_global_route_progress()
        self.interactive.phase = ""
        self.interactive.pending_command_id = -1
        self.interactive.pending_instruction = ""
        self.interactive.active_command_id = -1
        self.interactive.active_instruction = ""
        self.interactive.cancel_requested = False

    def reset_planner_control(self) -> None:
        self.trajectory.planner_control_mode = None
        self.trajectory.planner_yaw_delta_rad = None
        self.trajectory.planner_control_reason = ""
        self.trajectory.planner_control_queue = ()
        self.trajectory.planner_control_progress = 0.0
        self.trajectory.used_cached_traj = False

    def clear_global_route_progress(self) -> None:
        self.global_route.waypoints_world = []
        self.global_route.active_index = 0
        self.global_route.last_progress_ts = 0.0
        self.global_route.best_distance_m = float("inf")
        self.global_route.last_replan_reason = ""
        self.global_route.last_error = ""

    def interactive_overlay(self) -> dict[str, object]:
        return {
            "interactive_phase": str(self.interactive.phase),
            "interactive_command_id": int(self.interactive.active_command_id),
            "interactive_instruction": str(self.interactive.active_instruction),
        }

    def viewer_overlay_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "trajectory_world": np.asarray(self.trajectory.trajectory_world, dtype=np.float32).tolist(),
            "plan_version": int(self.trajectory.plan_version),
            "last_system2_signature": str(self.system2.last_signature),
            "last_system2_result": dict(self.system2.last_result),
            "last_system2_error": str(self.system2.last_error),
            "latest_system2_decision_mode": str(self.system2.latest_decision_mode),
            "navdp_last_request_goal_version": int(self.navdp.last_request_goal_version),
            "navdp_last_request_mode": str(self.navdp.last_request_mode),
            "navdp_last_request_started_at_s": float(self.navdp.last_request_started_at_s),
            "navdp_last_request_source_frame_id": int(self.navdp.last_request_source_frame_id),
            "navdp_last_committed_goal_version": int(self.navdp.last_committed_goal_version),
            "navdp_last_committed_plan_version": int(self.navdp.last_committed_plan_version),
            "navdp_last_discarded_goal_version": int(self.navdp.last_discarded_goal_version),
            "navdp_last_discard_reason": str(self.navdp.last_discard_reason),
            "navdp_request_active": bool(self.navdp.request_active),
            "navdp_error": str(self.navdp.error),
            "locomotion_state_label": str(self.locomotion.state_label),
            "locomotion_last_command": np.asarray(self.locomotion.last_command, dtype=np.float32).tolist(),
            "locomotion_last_command_stamp": float(self.locomotion.last_command_stamp),
        }
        if self.mode == "MEM_NAV" and (self.global_route.planner is not None or len(self.global_route.waypoints_world) > 0):
            route = self.global_route
            state.update(
                {
                    "global_route_enabled": True,
                    "global_route_active": bool(route.active_index < len(route.waypoints_world)),
                    "global_route_waypoint_index": int(route.active_index),
                    "global_route_waypoint_count": int(len(route.waypoints_world)),
                    "global_route_last_replan_reason": str(route.last_replan_reason),
                    "global_route_last_error": str(route.last_error),
                    "global_route_goal_xy": [float(route.final_goal_xy[0]), float(route.final_goal_xy[1])],
                    "global_route_waypoints_world": [
                        [float(waypoint[0]), float(waypoint[1])] for waypoint in route.waypoints_world
                    ],
                }
            )
            if route.active_index < len(route.waypoints_world):
                active_waypoint = route.waypoints_world[route.active_index]
                state["global_route_active_waypoint_xy"] = [float(active_waypoint[0]), float(active_waypoint[1])]
        if self.mode == "NAV":
            state.update(
                {
                    "goal_version": int(self.goal.goal_version),
                    "traj_version": int(self.goal.traj_version),
                    "stale_sec": float(self.trajectory.stale_sec),
                    "stale_hold_reason": str(self.trajectory.stale_hold_reason),
                    "planner_control_mode": self.trajectory.planner_control_mode,
                    "planner_yaw_delta_rad": self.trajectory.planner_yaw_delta_rad,
                    "planner_control_reason": str(self.trajectory.planner_control_reason),
                    "planner_control_queue": list(self.trajectory.planner_control_queue),
                    "planner_control_progress": float(self.trajectory.planner_control_progress),
                    "active_goal_mode": self.goal_mode(),
                    "active_goal_local_xy": _float_pair(self.goal.local_xy),
                    "active_goal_world_xy": _float_pair(self.goal.target_world_xy),
                    "active_pixel_goal": _int_pair(self.goal.target_pixel_xy),
                    "pending_goal_kind": str(self.goal.candidate_kind),
                    "pending_goal_world_xy": _float_pair(self.goal.filtered_candidate_world_xy),
                    "pending_pixel_goal": _int_pair(self.goal.filtered_candidate_pixel_xy),
                    "goal_candidate_sample_count": int(self.goal.candidate_sample_count),
                    "goal_candidate_started_at_s": float(self.goal.candidate_started_at_s),
                    "goal_candidate_last_stamp_s": float(self.goal.candidate_last_stamp_s),
                    "goal_last_clear_reason": str(self.goal.last_clear_reason),
                    "direct_action_mode": self.action_override.mode,
                    "direct_action_queue": list(self.action_override.pending_modes),
                    "direct_action_progress": float(self.action_override.progress),
                    "direct_action_started_at_s": float(self.action_override.started_at_s),
                    "direct_action_target_distance_m": float(self.action_override.target_distance_m),
                    "direct_action_target_yaw_rad": float(self.action_override.target_yaw_rad),
                }
            )
            if self.goal.system2_pixel_goal is not None:
                state["system2_pixel_goal"] = list(self.goal.system2_pixel_goal)
        return state
