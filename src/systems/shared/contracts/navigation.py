from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NavDpPlan:
    """Trajectory plan returned by the navigation subsystem."""

    trajectory_camera: np.ndarray
    all_trajectories_camera: np.ndarray | None
    values: np.ndarray | None
    plan_time_s: float
    stamp_s: float
