from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

from apps.isaacsim_runner.runner_config import DEFAULT_G1_GROUND_CLEARANCE_Z


@dataclass(frozen=True)
class StageReferenceSpec:
    usd_path: str
    prim_path: str
    kind: str = "reference"


@dataclass
class StageLayoutConfig:
    world_prim_path: str = "/World"
    environment_prim_path: str = "/World/Environment"
    robots_prim_path: str = "/World/Robots"
    physics_scene_prim_path: str = "/World/PhysicsScene"
    key_light_prim_path: str = "/World/Environment/KeyLight"
    key_light_intensity: float = 500.0
    key_light_angle: float = 0.53
    enable_flat_grid: bool = True
    flat_grid_prim_path: Optional[str] = None
    environment_refs: list[StageReferenceSpec] = field(default_factory=list)
    object_refs: list[StageReferenceSpec] = field(default_factory=list)


def _sanitize_stage_token(token: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", str(token)).strip("_")
    return sanitized or "Ref"


def _normalize_prim_path(path: str, fallback: str) -> str:
    value = str(path or "").strip()
    if not value:
        value = str(fallback or "").strip() or "/World"
    if not value.startswith("/"):
        value = f"/{value}"
    return value


def _split_reference_spec(spec: str) -> tuple[str, str]:
    # Format: "USD_PATH@/Prim/Path". If suffix after '@' does not look like a prim path,
    # keep the whole string as USD path (e.g. URIs containing user@host).
    if "@" not in spec:
        return spec, ""
    usd_candidate, prim_candidate = spec.rsplit("@", 1)
    if prim_candidate.strip().startswith("/"):
        return usd_candidate, prim_candidate
    return spec, ""


def _parse_stage_reference_specs(
    specs: Optional[Sequence[str]],
    default_parent_prim: str,
    default_prefix: str,
    kind: str,
) -> list[StageReferenceSpec]:
    parent_prim = _normalize_prim_path(default_parent_prim, "/World")
    out: list[StageReferenceSpec] = []
    for idx, raw in enumerate(specs or [], start=1):
        spec = str(raw or "").strip()
        if not spec:
            continue
        usd_part, prim_part = _split_reference_spec(spec)
        usd_path = str(usd_part).strip()
        if not usd_path:
            logging.warning("Skipping empty %s reference spec: %r", kind, raw)
            continue

        if str(prim_part).strip():
            prim_path = _normalize_prim_path(str(prim_part).strip(), f"{parent_prim}/{default_prefix}_{idx:02d}")
        else:
            stem = _sanitize_stage_token(Path(usd_path).stem or f"{default_prefix}_{idx:02d}")
            prim_path = f"{parent_prim.rstrip('/')}/{default_prefix}_{idx:02d}_{stem}"

        out.append(StageReferenceSpec(usd_path=usd_path, prim_path=prim_path, kind=kind))
    return out


def _ensure_parent_xforms(stage_obj, prim_path: str) -> None:
    try:
        from pxr import UsdGeom  # type: ignore
    except Exception:
        return

    tokens = [tok for tok in _normalize_prim_path(prim_path, "/World").split("/") if tok]
    current = ""
    for tok in tokens[:-1]:
        current += f"/{tok}"
        if not stage_obj.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage_obj, current)
            logging.info("Created Xform prim: %s", current)


def _move_prim_if_needed(stage_obj, path_from: str, path_to: str) -> bool:
    src = stage_obj.GetPrimAtPath(path_from)
    if not src.IsValid():
        return False

    norm_to = _normalize_prim_path(path_to, path_from)
    if path_from == norm_to:
        return False
    if stage_obj.GetPrimAtPath(norm_to).IsValid():
        return False

    _ensure_parent_xforms(stage_obj, norm_to)

    try:
        from pxr import Usd  # type: ignore

        namespace_editor = Usd.NamespaceEditor(stage_obj)
        namespace_editor.MovePrimAtPath(path_from, norm_to)
        if namespace_editor.ApplyEdits():
            logging.info("Moved prim: %s -> %s", path_from, norm_to)
            return True
    except Exception as exc:
        logging.debug("Usd.NamespaceEditor move failed (%s -> %s): %s", path_from, norm_to, exc)

    try:
        import omni.kit.commands  # type: ignore

        omni.kit.commands.execute(
            "MovePrim",
            path_from=path_from,
            path_to=norm_to,
            keep_world_transform=True,
        )
        logging.info("Moved prim with MovePrim command: %s -> %s", path_from, norm_to)
        return True
    except Exception as exc:
        logging.warning("Failed to move prim (%s -> %s): %s", path_from, norm_to, exc)
        return False


def _add_reference_to_stage_prim(stage_obj, usd_path: str, prim_path: str, label: str) -> bool:
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore
    except Exception as exc:
        logging.warning("Could not import stage utilities to add %s reference: %s", label, exc)
        return False

    target = _normalize_prim_path(prim_path, "/World")
    if stage_obj.GetPrimAtPath(target).IsValid():
        logging.info("%s reference target already exists at %s; skipping", label.capitalize(), target)
        return False

    _ensure_parent_xforms(stage_obj, target)

    try:
        add_reference_to_stage(usd_path, target)
        logging.info("Added %s reference: usd=%s, prim=%s", label, usd_path, target)
        return True
    except Exception as exc:
        logging.warning("Failed to add %s reference: usd=%s prim=%s err=%s", label, usd_path, target, exc)
        return False


def _add_flat_grid_environment(stage_obj, target_prim_path: Optional[str] = None) -> bool:
    targets: list[str] = []
    if target_prim_path:
        targets.append(_normalize_prim_path(target_prim_path, "/World/Environment/FlatGrid"))
    targets.extend(["/World/Environment/FlatGrid", "/World/FlatGrid", "/FlatGrid"])

    seen: set[str] = set()
    flat_grid_targets = []
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        flat_grid_targets.append(target)

    for target in flat_grid_targets:
        if stage_obj.GetPrimAtPath(target).IsValid():
            logging.info("Flat grid already present at %s", target)
            return False

    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore
    except Exception as exc:
        logging.warning("Could not import Isaac Sim assets helper for flat grid setup: %s", exc)
        return False

    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        logging.warning("Could not resolve Isaac Sim assets root. Skipping flat grid setup.")
        return False

    flat_grid_usd = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
    if target_prim_path:
        target_prim = _normalize_prim_path(target_prim_path, "/World/Environment/FlatGrid")
    elif stage_obj.GetPrimAtPath("/World/Environment").IsValid():
        target_prim = "/World/Environment/FlatGrid"
    elif stage_obj.GetPrimAtPath("/World").IsValid():
        target_prim = "/World/FlatGrid"
    else:
        target_prim = "/FlatGrid"

    return _add_reference_to_stage_prim(stage_obj, flat_grid_usd, target_prim, label="flat grid")


def _ensure_stage_hierarchy(stage_obj, world_path: str, environment_path: str, robots_path: str) -> None:
    try:
        from pxr import UsdGeom  # type: ignore
    except Exception:
        return

    for path in (
        _normalize_prim_path(world_path, "/World"),
        _normalize_prim_path(environment_path, "/World/Environment"),
        _normalize_prim_path(robots_path, "/World/Robots"),
    ):
        if not stage_obj.GetPrimAtPath(path).IsValid():
            UsdGeom.Xform.Define(stage_obj, path)
            logging.info("Created Xform prim: %s", path)


def _ensure_stage_default_prim(stage_obj, world_path: str) -> None:
    world_prim_path = _normalize_prim_path(world_path, "/World")
    world_prim = stage_obj.GetPrimAtPath(world_prim_path)
    if world_prim.IsValid() and stage_obj.GetDefaultPrim() != world_prim:
        stage_obj.SetDefaultPrim(world_prim)
        logging.info("Set default prim: %s", world_prim_path)


def _ensure_stage_physics_scene(stage_obj, physics_scene_path: str) -> None:
    try:
        from pxr import UsdPhysics  # type: ignore
    except Exception:
        return

    scene_path = _normalize_prim_path(physics_scene_path, "/World/PhysicsScene")
    if not stage_obj.GetPrimAtPath(scene_path).IsValid():
        _ensure_parent_xforms(stage_obj, scene_path)
        scene = UsdPhysics.Scene.Define(stage_obj, scene_path)
        scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        logging.info("Created default physics scene: %s", scene_path)


def _ensure_stage_key_light(stage_obj, light_path: str, intensity: float, angle: float) -> None:
    try:
        from pxr import UsdLux  # type: ignore
    except Exception:
        return

    light_prim_path = _normalize_prim_path(light_path, "/World/Environment/KeyLight")
    if not stage_obj.GetPrimAtPath(light_prim_path).IsValid():
        _ensure_parent_xforms(stage_obj, light_prim_path)
        light = UsdLux.DistantLight.Define(stage_obj, light_prim_path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateAngleAttr(float(angle))
        logging.info("Created default distant light: %s", light_prim_path)


def _ensure_world_environment(stage_obj, layout: Optional[StageLayoutConfig] = None) -> None:
    try:
        from pxr import Usd  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr modules for world setup: %s", exc)
        return

    cfg = layout or StageLayoutConfig()

    _ensure_stage_hierarchy(stage_obj, cfg.world_prim_path, cfg.environment_prim_path, cfg.robots_prim_path)
    _ensure_stage_default_prim(stage_obj, cfg.world_prim_path)
    _ensure_stage_physics_scene(stage_obj, cfg.physics_scene_prim_path)
    _ensure_stage_key_light(stage_obj, cfg.key_light_prim_path, cfg.key_light_intensity, cfg.key_light_angle)

    flat_grid_target = cfg.flat_grid_prim_path or f"{_normalize_prim_path(cfg.environment_prim_path, '/World/Environment').rstrip('/')}/FlatGrid"
    _move_prim_if_needed(stage_obj, "/FlatGrid", flat_grid_target)
    _move_prim_if_needed(stage_obj, "/World/FlatGrid", flat_grid_target)

    # Keep robot roots at their original authored paths.
    # Moving articulated roots with NamespaceEditor/MovePrim can leave stale
    # relationship targets on composed assets (especially bridge USDs), which
    # breaks physics stability.


def _apply_stage_references(stage_obj, refs: Sequence[StageReferenceSpec]) -> bool:
    changed = False
    for ref in refs:
        changed = _add_reference_to_stage_prim(stage_obj, ref.usd_path, ref.prim_path, ref.kind) or changed
    return changed


def _apply_stage_layout(stage_obj, layout: StageLayoutConfig) -> bool:
    _ensure_world_environment(stage_obj, layout)

    changed = False
    if layout.enable_flat_grid:
        changed = _add_flat_grid_environment(stage_obj, target_prim_path=layout.flat_grid_prim_path) or changed

    changed = _apply_stage_references(stage_obj, layout.environment_refs) or changed
    changed = _apply_stage_references(stage_obj, layout.object_refs) or changed

    if changed:
        logging.info(
            "Applied stage layout: flat_grid=%s env_refs=%d object_refs=%d",
            bool(layout.enable_flat_grid),
            len(layout.environment_refs),
            len(layout.object_refs),
        )
    return changed


def _quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
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


def _read_base_pose(stage_obj, robot_prim_path: str) -> Optional[Dict[str, float]]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception:
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
        roll, pitch, yaw = _quat_to_rpy(qw, qx, qy, qz)
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


def _get_translate_op(xform) -> Optional[object]:
    try:
        from pxr import UsdGeom  # type: ignore
    except Exception:
        return None

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return None


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


def _resolve_robot_placement_prim(stage_obj, robot_prim_path: str):
    prim = stage_obj.GetPrimAtPath(robot_prim_path)
    if not prim.IsValid():
        return prim

    try:
        from pxr import UsdGeom, UsdPhysics  # type: ignore
    except Exception:
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


def _rebase_world_anchor_articulation_root(stage_obj, robot_prim_path: str) -> str:
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

    placement_prim = _resolve_robot_placement_prim(stage_obj, robot_prim_path)
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


def _ensure_robot_dynamic_flags(stage_obj, robot_prim_path: str) -> None:
    try:
        from pxr import UsdPhysics  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr UsdPhysics to normalize dynamic flags: %s", exc)
        return

    placement_prim = _resolve_robot_placement_prim(stage_obj, robot_prim_path)
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


def _find_motion_root_prim(stage_obj, robot_prim_path: str) -> str:
    if not robot_prim_path:
        return robot_prim_path
    prim = _resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if prim.IsValid():
        return prim.GetPath().pathString
    return robot_prim_path


def _find_robot_prim_path(stage_obj) -> Optional[str]:
    try:
        from pxr import UsdPhysics
    except Exception:
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


def _set_robot_start_height(stage_obj, robot_prim_path: str, z: float) -> None:
    try:
        from pxr import Gf, Usd, UsdGeom  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr modules to set robot start height: %s", exc)
        return

    prim = _resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if not prim.IsValid():
        logging.warning("Robot prim is not valid; skip start height set: %s", robot_prim_path)
        return

    if not prim.IsA(UsdGeom.Xformable):
        logging.warning("Robot prim is not xformable; skip start height set: %s", robot_prim_path)
        return

    xform = UsdGeom.Xformable(prim)
    translate_op = _get_translate_op(xform)
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


def _find_camera_prim_path(stage_obj) -> Optional[str]:
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
