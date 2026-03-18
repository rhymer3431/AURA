from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str = ""
    instruction: str = ""
    mode: str = ""
    state: str = "idle"
    command_id: int = -1


@dataclass(frozen=True)
class WorldStateSnapshot:
    current_task: TaskSnapshot = field(default_factory=TaskSnapshot)
    mode: str = ""
    robot_pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_yaw_rad: float = 0.0
    last_perception_summary: dict[str, Any] = field(default_factory=dict)
    last_memory_context: dict[str, Any] = field(default_factory=dict)
    last_s2_result: dict[str, Any] = field(default_factory=dict)
    active_nav_plan: dict[str, Any] = field(default_factory=dict)
    recovery_state: dict[str, Any] = field(default_factory=dict)
    stale_timers: dict[str, Any] = field(default_factory=dict)
    last_command_decision: dict[str, Any] = field(default_factory=dict)
