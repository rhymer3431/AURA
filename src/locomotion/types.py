from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
