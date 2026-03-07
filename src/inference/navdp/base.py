from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class NavDPPointGoalResponse:
    trajectory: np.ndarray
    all_trajectory: np.ndarray
    all_values: np.ndarray
    server_input_meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class NavDPNoGoalResponse:
    trajectory: np.ndarray
    all_trajectory: np.ndarray
    all_values: np.ndarray


class NavDPExecutionClient(Protocol):
    def navigator_reset(self, intrinsic: np.ndarray, batch_size: int = 1) -> str:
        ...

    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
        sensor_meta: dict[str, Any] | None = None,
    ) -> NavDPPointGoalResponse:
        ...

    def nogoal_step(
        self,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
    ) -> NavDPNoGoalResponse:
        ...
