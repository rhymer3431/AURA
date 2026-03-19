from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ipc.messages import ActionCommand

from .execution_mode import ExecutionMode
from .events import WorkerMetadata
from .world_state import TaskSnapshot


@dataclass(frozen=True)
class PlanningContext:
    metadata: WorkerMetadata
    task: TaskSnapshot
    planner_mode: ExecutionMode
    instruction: str
    robot_pose_xyz: tuple[float, float, float]
    robot_yaw_rad: float
    current_goal: tuple[float, float, float] | None = None
    perception_summary: dict[str, Any] = field(default_factory=dict)
    memory_summary: dict[str, Any] = field(default_factory=dict)
    manual_command: ActionCommand | None = None
