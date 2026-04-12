"""Helpers for choosing which camera prim is controlled by the pitch API."""

from __future__ import annotations


def resolve_camera_control_prim_path(robot_prim_path: str, camera_prim_path: str | None) -> str:
    """Prefer the existing D455 rig and fall back to a runtime camera prim."""

    if camera_prim_path:
        return str(camera_prim_path)

    from isaacsim.core.utils.prims import get_prim_at_path

    realsense_root = f"{robot_prim_path}/head_link/Realsense"
    if get_prim_at_path(realsense_root).IsValid():
        return realsense_root
    return f"{robot_prim_path}/NavCamera"
