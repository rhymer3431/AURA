"""Scene helpers for the standalone G1 ONNX runner."""

from __future__ import annotations

from typing import Optional


def spawn_environment(reference_path: Optional[str], prim_path: str, translation: tuple[float, float, float]):
    from isaacsim.core.utils.prims import define_prim

    if not reference_path:
        print("[WARN] No environment USD found. Running with an empty stage.")
        return

    prim = define_prim(prim_path, "Xform")
    prim.GetReferences().AddReference(reference_path)

    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            translate_op = op
            break

    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
