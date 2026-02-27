from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from isaacsim import SimulationApp


USD_PATHS = [
    Path(r"C:/Users/mango/project/isaac-aura/g1/g1_d455.usd"),
    Path(r"C:/Users/mango/project/isaac-aura/apps/isaac_ros2_bridge_bundle/robot_model/model_data/g1/g1_29dof_with_hand/g1_29dof_with_hand.usd"),
]


def _count_rigid_flags(stage, prefix_prim, usd_physics) -> dict[str, int]:
    if not prefix_prim.IsValid():
        return {"rigid_bodies": 0, "kinematic_true": 0, "disable_gravity_true": 0}

    prefix = prefix_prim.GetPath()
    rigid_bodies = 0
    kinematic_true = 0
    disable_gravity_true = 0

    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(prefix):
            continue
        if not prim.HasAPI(usd_physics.RigidBodyAPI):
            continue

        rigid_bodies += 1

        for name in ("physics:kinematicEnabled", "physxRigidBody:kinematicEnabled"):
            attr = prim.GetAttribute(name)
            if not attr.IsValid():
                continue
            try:
                if bool(attr.Get()):
                    kinematic_true += 1
                    break
            except Exception:
                pass

        for name in ("physxRigidBody:disableGravity", "physics:disableGravity"):
            attr = prim.GetAttribute(name)
            if not attr.IsValid():
                continue
            try:
                if bool(attr.Get()):
                    disable_gravity_true += 1
                    break
            except Exception:
                pass

    return {
        "rigid_bodies": rigid_bodies,
        "kinematic_true": kinematic_true,
        "disable_gravity_true": disable_gravity_true,
    }


def inspect_one(sim_app, usd_path: Path) -> dict[str, Any]:
    import omni.usd  # type: ignore
    from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore

    context = omni.usd.get_context()
    ok = context.open_stage(str(usd_path))
    if not ok:
        return {"usd": str(usd_path), "error": "open_stage failed"}

    for _ in range(120):
        sim_app.update()

    stage = context.get_stage()
    roots: list[str] = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(prim.GetPath().pathString)
    roots.sort()

    root_infos = []
    for root_path in roots:
        root_prim = stage.GetPrimAtPath(root_path)
        parent_prim = root_prim.GetParent()
        parent_path = parent_prim.GetPath().pathString if parent_prim.IsValid() else ""

        root_has_robot_joints = bool(root_prim.GetAttribute("isaac:physics:robotJoints").IsValid())
        parent_has_robot_joints = bool(
            parent_prim.IsValid() and parent_prim.GetAttribute("isaac:physics:robotJoints").IsValid()
        )

        bbox_root = None
        bbox_parent = None
        try:
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
            )
            root_range = cache.ComputeWorldBound(root_prim).ComputeAlignedRange()
            if not root_range.IsEmpty():
                bbox_root = {
                    "min": [float(v) for v in root_range.GetMin()],
                    "max": [float(v) for v in root_range.GetMax()],
                }
            if parent_prim.IsValid():
                parent_range = cache.ComputeWorldBound(parent_prim).ComputeAlignedRange()
                if not parent_range.IsEmpty():
                    bbox_parent = {
                        "min": [float(v) for v in parent_range.GetMin()],
                        "max": [float(v) for v in parent_range.GetMax()],
                    }
        except Exception as exc:
            bbox_root = {"error": str(exc)}

        root_infos.append(
            {
                "root": root_path,
                "parent": parent_path,
                "root_has_robot_joints": root_has_robot_joints,
                "parent_has_robot_joints": parent_has_robot_joints,
                "bbox_root": bbox_root,
                "bbox_parent": bbox_parent,
                "flags_root": _count_rigid_flags(stage, root_prim, UsdPhysics),
                "flags_parent": _count_rigid_flags(stage, parent_prim, UsdPhysics),
            }
        )

    return {
        "usd": str(usd_path),
        "articulation_roots": roots,
        "root_infos": root_infos,
    }


def main() -> None:
    sim_app = SimulationApp({"headless": True})
    try:
        results = [inspect_one(sim_app, p) for p in USD_PATHS]
        out_path = Path(r"C:/Users/mango/project/isaac-aura/tmp/inspect_usd_dynamics.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
