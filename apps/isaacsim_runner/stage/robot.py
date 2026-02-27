from __future__ import annotations

import logging
from typing import Optional

from apps.isaacsim_runner.config.base import DEFAULT_G1_GROUND_CLEARANCE_Z
from apps.isaacsim_runner.stage.prims import get_translate_op


def _count_rigid_bodies_under(stage_obj, prim, usd_physics) -> int:
    if not prim.IsValid():
        return 0
    prefix = prim.GetPath()
    count = 0
    for p in stage_obj.Traverse():
        if not p.GetPath().HasPrefix(prefix):
            continue
        if p.HasAPI(usd_physics.RigidBodyAPI):
            count += 1
    return count


def resolve_robot_placement_prim(stage_obj, robot_prim_path: str):
    prim = stage_obj.GetPrimAtPath(robot_prim_path)
    if not prim.IsValid():
        return prim

    try:
        from pxr import UsdGeom, UsdPhysics  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr modules to resolve robot placement prim: %s", exc)
        # Fallback: If the articulation root path points to a link (e.g. pelvis), move the parent model prim.
        if not prim.GetAttribute("isaac:physics:robotJoints").IsValid():
            parent = prim.GetParent()
            if parent.IsValid() and parent.GetPath().pathString != "/":
                return parent
        return prim

    # Prefer a xformable ancestor that actually owns the articulated rigid bodies.
    # Some USDs report articulation root on a joint prim (e.g. ".../root_joint"), which has no rigid bodies.
    if prim.IsA(UsdGeom.Xformable):
        self_count = _count_rigid_bodies_under(stage_obj, prim, UsdPhysics)
        if self_count >= 5:
            return prim

    parent = prim.GetParent()
    while parent.IsValid() and parent.GetPath().pathString not in ("", "/"):
        if parent.IsA(UsdGeom.Xformable):
            parent_count = _count_rigid_bodies_under(stage_obj, parent, UsdPhysics)
            if parent_count >= 5:
                return parent
        parent = parent.GetParent()

    return prim


def rebase_world_anchor_articulation_root(stage_obj, robot_prim_path: str) -> str:
    try:
        from pxr import PhysxSchema, Sdf, UsdPhysics  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr modules to inspect/rebase articulation root: %s", exc)
        return robot_prim_path

    root_prim = stage_obj.GetPrimAtPath(robot_prim_path)
    if not root_prim.IsValid() or root_prim.GetTypeName() != "PhysicsFixedJoint":
        return robot_prim_path
    if not root_prim.IsA(UsdPhysics.Joint):
        return robot_prim_path

    body0_rel = root_prim.GetRelationship("physics:body0")
    body1_rel = root_prim.GetRelationship("physics:body1")
    body0_targets = [str(path) for path in body0_rel.GetTargets()] if body0_rel and body0_rel.IsValid() else []
    body1_targets = [str(path) for path in body1_rel.GetTargets()] if body1_rel and body1_rel.IsValid() else []
    world_anchor = (len(body0_targets) == 0) or all(target in ("/World", "/world") for target in body0_targets)
    if (not world_anchor) or (not body1_targets):
        return robot_prim_path

    placement_prim = resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if not placement_prim.IsValid():
        return robot_prim_path

    enabled_attr = root_prim.GetAttribute("physics:jointEnabled")
    if not enabled_attr.IsValid():
        enabled_attr = root_prim.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool)
    enabled_attr.Set(False)

    try:
        root_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    except Exception:
        pass
    try:
        root_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    except Exception:
        pass

    if not placement_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(placement_prim)
    if not placement_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(placement_prim)

    rebased_path = placement_prim.GetPath().pathString
    logging.info(
        "Rebased world-anchor articulation root for floating base: old=%s new=%s body1=%s",
        robot_prim_path,
        rebased_path,
        body1_targets,
    )
    return rebased_path


def ensure_robot_dynamic_flags(stage_obj, robot_prim_path: str) -> None:
    try:
        from pxr import UsdPhysics  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr UsdPhysics to normalize dynamic flags: %s", exc)
        return

    placement_prim = resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if not placement_prim.IsValid():
        return

    root_path = placement_prim.GetPath()
    rigid_bodies = 0
    kinematic_fixed = 0
    gravity_fixed = 0

    for prim in stage_obj.Traverse():
        if not prim.GetPath().HasPrefix(root_path):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_bodies += 1

        for attr_name in ("physics:kinematicEnabled", "physxRigidBody:kinematicEnabled"):
            attr = prim.GetAttribute(attr_name)
            if not attr.IsValid():
                continue
            try:
                if bool(attr.Get()):
                    attr.Set(False)
                    kinematic_fixed += 1
            except Exception:
                pass

        for attr_name in ("physxRigidBody:disableGravity", "physics:disableGravity"):
            attr = prim.GetAttribute(attr_name)
            if not attr.IsValid():
                continue
            try:
                if bool(attr.Get()):
                    attr.Set(False)
                    gravity_fixed += 1
            except Exception:
                pass

    if rigid_bodies > 0 and (kinematic_fixed > 0 or gravity_fixed > 0):
        logging.info(
            "Normalized rigid body flags under %s: rigid_bodies=%d kinematic_disabled=%d gravity_enabled=%d",
            placement_prim.GetPath().pathString,
            rigid_bodies,
            kinematic_fixed,
            gravity_fixed,
        )


def find_motion_root_prim(stage_obj, robot_prim_path: str) -> str:
    if not robot_prim_path:
        return robot_prim_path
    prim = resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if prim.IsValid():
        return prim.GetPath().pathString
    return robot_prim_path


def find_robot_prim_path(stage_obj) -> Optional[str]:
    try:
        from pxr import UsdPhysics
    except Exception as exc:
        logging.warning("Could not import pxr UsdPhysics for robot prim detection: %s", exc)
        return None

    articulation_roots = []
    for prim in stage_obj.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim.GetPath().pathString)
    if articulation_roots:
        # Prefer the shortest path (closest to stage root) if there are multiple roots.
        articulation_roots.sort(key=lambda p: (p.count("/"), len(p)))
        selected = articulation_roots[0]
        logging.info("Detected articulation root prim(s): %s -> selected=%s", articulation_roots, selected)
        return selected

    preferred = (
        "/World/Robots/g1_29dof_with_hand_rev_1_0",
        "/g1_29dof_with_hand_rev_1_0",
        "/World/g1_29dof_with_hand_rev_1_0",
        "/World/Robots/g1",
        "/g1",
        "/World/g1",
    )
    for path in preferred:
        prim = stage_obj.GetPrimAtPath(path)
        if prim.IsValid():
            logging.info(
                "Falling back to preferred robot prim path without ArticulationRootAPI: %s",
                path,
            )
            return path

    for prim in stage_obj.Traverse():
        if prim.GetAttribute("isaac:physics:robotJoints").IsValid():
            path = prim.GetPath().pathString
            logging.info(
                "Falling back to robotJoints attribute prim path without ArticulationRootAPI: %s",
                path,
            )
            return path

    return None


def set_robot_start_height(stage_obj, robot_prim_path: str, z: float) -> None:
    try:
        from pxr import Gf, Usd, UsdGeom  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr modules to set robot start height: %s", exc)
        return

    prim = resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if not prim.IsValid():
        logging.warning("Robot prim is not valid; skip start height set: %s", robot_prim_path)
        return

    if not prim.IsA(UsdGeom.Xformable):
        logging.warning("Robot prim is not xformable; skip start height set: %s", robot_prim_path)
        return

    xform = UsdGeom.Xformable(prim)
    translate_op = get_translate_op(xform)
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    current = translate_op.Get() or Gf.Vec3d(0.0, 0.0, z)
    try:
        x_pos = float(current[0])
        y_pos = float(current[1])
        z_pos = float(current[2])
    except Exception:
        x_pos = 0.0
        y_pos = 0.0
        z_pos = z

    aligned = None
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        )
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    except Exception as exc:
        logging.warning("Failed to compute robot world bounds for spawn placement: %s", exc)

    if aligned is None or aligned.IsEmpty():
        translate_op.Set((x_pos, y_pos, z))
        logging.info(
            "Set robot start height via fallback: prim=%s z=%.3f (bounds unavailable)",
            prim.GetPath().pathString,
            z,
        )
        return

    min_z = float(aligned.GetMin()[2])
    lift_z = max(0.0, DEFAULT_G1_GROUND_CLEARANCE_Z - min_z)
    target_z = z_pos + lift_z
    translate_op.Set((x_pos, y_pos, target_z))
    logging.info(
        "Adjusted robot spawn clearance: prim=%s min_z=%.3f clearance=%.3f lift=%.3f final_z=%.3f",
        prim.GetPath().pathString,
        min_z,
        DEFAULT_G1_GROUND_CLEARANCE_Z,
        lift_z,
        target_z,
    )


def find_camera_prim_path(stage_obj) -> Optional[str]:
    camera_paths = []
    for prim in stage_obj.Traverse():
        if prim.GetTypeName() == "Camera":
            camera_paths.append(prim.GetPath().pathString)
    if not camera_paths:
        return None

    # Prefer physical sensor camera prims, especially color camera.
    for path in camera_paths:
        lowered = path.lower()
        if "color" in lowered and "omniversekit_" not in lowered:
            return path
    for path in camera_paths:
        lowered = path.lower()
        if ("d435" in lowered or "rsd455" in lowered or "camera" in lowered) and "omniversekit_" not in lowered:
            return path
    return camera_paths[0]
