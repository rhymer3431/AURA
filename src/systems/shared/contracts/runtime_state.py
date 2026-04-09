from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .inference import System2Result
from .navigation import NavDpPlan
from .planner import Subgoal, TaskFrame


def _zero_command() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


def _empty_world_path() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float32)


@dataclass(slots=True)
class PlannerInput:
    """Snapshot passed from world-state capture into planning."""

    robot_state: Any
    goal_xy_body: np.ndarray | None
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    camera_pos_w: np.ndarray
    camera_rot_w: np.ndarray
    stamp_s: float


@dataclass(slots=True)
class CommandState:
    instruction: str
    language: str
    command_revision: int = 0


@dataclass(slots=True)
class CaptureState:
    latest_input: PlannerInput | None = None
    last_capture_time: float = 0.0


@dataclass(slots=True)
class System2RuntimeState:
    session_reset_required: bool = True
    last_input_stamp: float = 0.0
    last_signature: tuple[str, tuple[float, ...] | None, str] | None = None
    last_result: System2Result | None = None
    error: str | None = None


@dataclass(slots=True)
class GoalState:
    tolerance: float
    target_mode: str = "none"
    target_world_xy: np.ndarray | None = None
    target_pixel_xy: np.ndarray | None = None
    last_update_stamp_s: float = 0.0
    last_clear_reason: str = "init"
    generation: int = 0
    candidate_kind: str = "none"
    raw_candidate_world_xy: np.ndarray | None = None
    filtered_candidate_world_xy: np.ndarray | None = None
    raw_candidate_pixel_xy: np.ndarray | None = None
    filtered_candidate_pixel_xy: np.ndarray | None = None
    candidate_started_at_s: float = 0.0
    candidate_last_stamp_s: float = 0.0
    candidate_sample_count: int = 0


@dataclass(slots=True)
class NavDpState:
    algorithm_name: str | None = None
    last_input_stamp: float = 0.0
    last_goal_generation: int = -1
    last_request_input_stamp: float = 0.0
    last_request_goal_generation: int = -1
    last_request_started_at_s: float = 0.0
    last_committed_goal_generation: int = -1
    last_discarded_goal_generation: int = -1
    last_discard_reason: str | None = None
    latest_plan: NavDpPlan | None = None
    latest_world_path: np.ndarray = field(default_factory=_empty_world_path)
    latest_plan_time: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class ActionOverrideState:
    mode: str | None = None
    pending_modes: tuple[str, ...] = ()
    started_at_s: float = 0.0
    start_pos_xy: np.ndarray | None = None
    start_yaw: float = 0.0
    target_distance_m: float = 0.0
    target_yaw_rad: float = 0.0
    progress: float = 0.0


@dataclass(slots=True)
class LocomotionState:
    command: np.ndarray = field(default_factory=_zero_command)
    state_label: str = "waiting"
    last_command_stamp: float = 0.0


@dataclass(slots=True)
class StatusState:
    last_warning_time: float = 0.0
    last_status_time: float = 0.0
    goal_done_reported: bool = False
    action_only_suppressed: bool = False
    last_action_only_mode: str | None = None


@dataclass(slots=True)
class NavigationPipelineState:
    command: CommandState
    capture: CaptureState = field(default_factory=CaptureState)
    system2: System2RuntimeState = field(default_factory=System2RuntimeState)
    goal: GoalState | None = None
    navdp: NavDpState = field(default_factory=NavDpState)
    action_override: ActionOverrideState = field(default_factory=ActionOverrideState)
    follower_smoothed_cmd: np.ndarray = field(default_factory=_zero_command)
    follower_last_time: float = 0.0
    locomotion: LocomotionState = field(default_factory=LocomotionState)
    status: StatusState = field(default_factory=StatusState)


@dataclass(slots=True)
class TaskExecutionState:
    task_id: str
    raw_instruction: str
    language: str
    task_frame: TaskFrame
    subgoals: list[Subgoal]
    current_subgoal_index: int = 0
    origin_pose: dict[str, object] | None = None
    status: str = "pending"
    last_report: str | None = None
    failure_reason: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None
