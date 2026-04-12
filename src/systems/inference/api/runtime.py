"""Runtime-facing inference facade."""

from systems.inference.client import (
    InternVlaNavClient,
    normalized_uv_to_pixel_xy,
    pixel_xy_to_normalized_uv,
    resolve_goal_world_xy,
    resolve_goal_world_xy_from_pixel,
)
from systems.inference.stack.config import ManagedServiceConfig, REPO_ROOT
from systems.inference.stack.process_registry import ProcessRegistry
from systems.shared.contracts.inference import System2Result

__all__ = [
    "InternVlaNavClient",
    "ManagedServiceConfig",
    "ProcessRegistry",
    "REPO_ROOT",
    "System2Result",
    "normalized_uv_to_pixel_xy",
    "pixel_xy_to_normalized_uv",
    "resolve_goal_world_xy",
    "resolve_goal_world_xy_from_pixel",
]
