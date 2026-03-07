from __future__ import annotations

import os
import re

import numpy as np


def get_single_pose(xform_prim) -> tuple[np.ndarray, np.ndarray]:
    positions, orientations = xform_prim.get_world_poses()
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)[0]
    quat = np.asarray(orientations, dtype=np.float32).reshape(-1, 4)[0]
    return pos.copy(), quat.copy()


def disable_rigid_bodies(stage, root_path: str) -> int:
    disabled = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root_path):
            continue
        attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if not attr.IsValid():
            continue
        try:
            attr.Set(False)
            disabled += 1
        except Exception:  # noqa: BLE001
            continue
    return disabled


def resolve_environment_reference(env_url: str, assets_root_path: str) -> str:
    raw = str(env_url).strip()
    if raw == "":
        return ""

    lower_raw = raw.lower()
    if "://" in raw or lower_raw.startswith("file:"):
        return raw

    if re.match(r"^[A-Za-z]:[\\/]", raw) or raw.startswith("\\\\"):
        return os.path.normpath(raw)

    normalized_raw = os.path.normpath(raw)
    if os.path.exists(normalized_raw):
        return os.path.abspath(normalized_raw)

    return assets_root_path + raw


def place_goal_marker(
    stage,
    goal_world_xy: np.ndarray,
    *,
    marker_root_path: str = "/World/GoalMarker",
    marker_radius_m: float = 0.12,
    marker_z_m: float = 0.12,
) -> tuple[bool, str]:
    try:
        from pxr import Gf, UsdGeom
    except Exception as exc:  # noqa: BLE001
        return False, f"pxr import failed: {type(exc).__name__}: {exc}"

    try:
        root_prim = stage.DefinePrim(marker_root_path, "Xform")
        marker_prim_path = f"{marker_root_path}/Point"
        sphere_prim = stage.DefinePrim(marker_prim_path, "Sphere")
        if not root_prim.IsValid() or not sphere_prim.IsValid():
            return False, "failed to define marker prims"

        sphere = UsdGeom.Sphere(sphere_prim)
        sphere.CreateRadiusAttr(float(marker_radius_m))
        gprim = UsdGeom.Gprim(sphere_prim)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.15, 0.1)])

        root_xform = UsdGeom.Xformable(root_prim)
        translate_op = None
        for op in root_xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = root_xform.AddTranslateOp()
        translate_op.Set(
            Gf.Vec3d(
                float(goal_world_xy[0]),
                float(goal_world_xy[1]),
                float(marker_z_m),
            )
        )
        return True, marker_prim_path
    except Exception as exc:  # noqa: BLE001
        return False, f"goal marker creation failed: {type(exc).__name__}: {exc}"
