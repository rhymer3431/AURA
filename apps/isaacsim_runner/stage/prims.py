from __future__ import annotations

import logging
import math
from typing import Dict, Optional


def normalize_prim_path(path: str, fallback: str) -> str:
    value = str(path or "").strip()
    if not value:
        value = str(fallback or "").strip() or "/World"
    if not value.startswith("/"):
        value = f"/{value}"
    return value


def ensure_parent_xforms(stage_obj, prim_path: str) -> None:
    try:
        from pxr import UsdGeom  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr UsdGeom for parent xform setup: %s", exc)
        return

    tokens = [tok for tok in normalize_prim_path(prim_path, "/World").split("/") if tok]
    current = ""
    for tok in tokens[:-1]:
        current += f"/{tok}"
        if not stage_obj.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage_obj, current)
            logging.info("Created Xform prim: %s", current)


def quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def read_base_pose(stage_obj, robot_prim_path: str) -> Optional[Dict[str, float]]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr modules to read base pose: %s", exc)
        return None

    prim = stage_obj.GetPrimAtPath(robot_prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
        return None
    try:
        world_t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        trans = world_t.ExtractTranslation()
        quat = world_t.ExtractRotationQuat()
        imag = quat.GetImaginary()
        qx = float(imag[0])
        qy = float(imag[1])
        qz = float(imag[2])
        qw = float(quat.GetReal())
        roll, pitch, yaw = quat_to_rpy(qw, qx, qy, qz)
        return {
            "x": float(trans[0]),
            "y": float(trans[1]),
            "height": float(trans[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        }
    except Exception:
        return None


def get_translate_op(xform) -> Optional[object]:
    try:
        from pxr import UsdGeom  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr UsdGeom for translate op lookup: %s", exc)
        return None

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return None
