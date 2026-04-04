from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from runtime.planning_session import PlannerStats
from schemas.execution_mode import ExecutionMode, normalize_execution_mode


@dataclass
class GoalState:
    local_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    goal_version: int = -1
    traj_version: int = -1
    system2_result_version: int = -1
    system2_pixel_goal: list[int] | None = None
    system2_submit_ts: float = 0.0
    system2_response_ts: float = field(default_factory=time.perf_counter)
    nav_instruction: str = ""


@dataclass
class TrajectoryState:
    trajectory_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    plan_version: int = -1
    last_plan_stamp_s: float = 0.0
    planner_control_mode: str | None = None
    planner_yaw_delta_rad: float | None = None
    planner_control_reason: str = ""
    used_cached_traj: bool = False
    stale_sec: float = -1.0
    stats: PlannerStats = field(default_factory=PlannerStats)


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
    final_goal_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
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

    def reset_navigation_state(self) -> None:
        self.goal.local_xy = np.zeros(2, dtype=np.float32)
        self.goal.goal_version = -1
        self.goal.traj_version = -1
        self.goal.system2_result_version = -1
        self.goal.system2_pixel_goal = None
        self.goal.system2_submit_ts = 0.0
        self.goal.system2_response_ts = time.perf_counter()
        self.goal.nav_instruction = ""
        self.trajectory.trajectory_world = np.zeros((0, 3), dtype=np.float32)
        self.trajectory.plan_version = -1
        self.trajectory.last_plan_stamp_s = 0.0
        self.trajectory.stale_sec = -1.0
        self.trajectory.stats = PlannerStats()
        self.reset_planner_control()
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
                    "planner_control_mode": self.trajectory.planner_control_mode,
                    "planner_yaw_delta_rad": self.trajectory.planner_yaw_delta_rad,
                    "planner_control_reason": str(self.trajectory.planner_control_reason),
                }
            )
            if self.goal.system2_pixel_goal is not None:
                state["system2_pixel_goal"] = list(self.goal.system2_pixel_goal)
        return state
