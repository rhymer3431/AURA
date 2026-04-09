"""Runtime-facing inference facade."""

from systems.inference.infrastructure.internvla_nav import (
    InternVlaNavClient,
    normalized_uv_to_pixel_xy,
    pixel_xy_to_normalized_uv,
    resolve_goal_world_xy,
    resolve_goal_world_xy_from_pixel,
)
from systems.shared.contracts.inference import System2Result

__all__ = [
    "InternVlaNavClient",
    "System2Result",
    "normalized_uv_to_pixel_xy",
    "pixel_xy_to_normalized_uv",
    "resolve_goal_world_xy",
    "resolve_goal_world_xy_from_pixel",
]
