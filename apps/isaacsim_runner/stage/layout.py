from __future__ import annotations

import logging
from typing import Optional, Sequence

from apps.isaacsim_runner.config.stage import StageLayoutConfig, StageReferenceSpec
from apps.isaacsim_runner.stage.prims import ensure_parent_xforms, normalize_prim_path


def _move_prim_if_needed(stage_obj, path_from: str, path_to: str) -> bool:
    src = stage_obj.GetPrimAtPath(path_from)
    if not src.IsValid():
        return False

    norm_to = normalize_prim_path(path_to, path_from)
    if path_from == norm_to:
        return False
    if stage_obj.GetPrimAtPath(norm_to).IsValid():
        return False

    ensure_parent_xforms(stage_obj, norm_to)

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

    target = normalize_prim_path(prim_path, "/World")
    if stage_obj.GetPrimAtPath(target).IsValid():
        logging.info("%s reference target already exists at %s; skipping", label.capitalize(), target)
        return False

    ensure_parent_xforms(stage_obj, target)

    try:
        add_reference_to_stage(usd_path, target)
        logging.info("Added %s reference: usd=%s, prim=%s", label, usd_path, target)
        return True
    except Exception as exc:
        logging.warning("Failed to add %s reference: usd=%s prim=%s err=%s", label, usd_path, target, exc)
        return False


def _add_flat_grid(stage_obj, target_prim_path: Optional[str] = None) -> bool:
    targets: list[str] = []
    if target_prim_path:
        targets.append(normalize_prim_path(target_prim_path, "/World/Environment/FlatGrid"))
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
        target_prim = normalize_prim_path(target_prim_path, "/World/Environment/FlatGrid")
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
    except Exception as exc:
        logging.warning("Could not import pxr UsdGeom for stage hierarchy setup: %s", exc)
        return

    for path in (
        normalize_prim_path(world_path, "/World"),
        normalize_prim_path(environment_path, "/World/Environment"),
        normalize_prim_path(robots_path, "/World/Robots"),
    ):
        if not stage_obj.GetPrimAtPath(path).IsValid():
            UsdGeom.Xform.Define(stage_obj, path)
            logging.info("Created Xform prim: %s", path)


def _ensure_stage_default_prim(stage_obj, world_path: str) -> None:
    world_prim_path = normalize_prim_path(world_path, "/World")
    world_prim = stage_obj.GetPrimAtPath(world_prim_path)
    if world_prim.IsValid() and stage_obj.GetDefaultPrim() != world_prim:
        stage_obj.SetDefaultPrim(world_prim)
        logging.info("Set default prim: %s", world_prim_path)


def _ensure_stage_physics_scene(stage_obj, physics_scene_path: str) -> None:
    try:
        from pxr import UsdPhysics  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr UsdPhysics for stage physics scene setup: %s", exc)
        return

    scene_path = normalize_prim_path(physics_scene_path, "/World/PhysicsScene")
    if not stage_obj.GetPrimAtPath(scene_path).IsValid():
        ensure_parent_xforms(stage_obj, scene_path)
        scene = UsdPhysics.Scene.Define(stage_obj, scene_path)
        scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        logging.info("Created default physics scene: %s", scene_path)


def _ensure_stage_key_light(stage_obj, light_path: str, intensity: float, angle: float) -> None:
    try:
        from pxr import UsdLux  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr UsdLux for stage key light setup: %s", exc)
        return

    light_prim_path = normalize_prim_path(light_path, "/World/Environment/KeyLight")
    if not stage_obj.GetPrimAtPath(light_prim_path).IsValid():
        ensure_parent_xforms(stage_obj, light_prim_path)
        light = UsdLux.DistantLight.Define(stage_obj, light_prim_path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateAngleAttr(float(angle))
        logging.info("Created default distant light: %s", light_prim_path)


def ensure_world_environment(stage_obj, layout: Optional[StageLayoutConfig] = None) -> None:
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

    flat_grid_target = cfg.flat_grid_prim_path or f"{normalize_prim_path(cfg.environment_prim_path, '/World/Environment').rstrip('/')}/FlatGrid"
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


def apply_stage_layout(stage_obj, layout: StageLayoutConfig) -> bool:
    ensure_world_environment(stage_obj, layout)

    changed = False
    if layout.enable_flat_grid:
        changed = _add_flat_grid(stage_obj, target_prim_path=layout.flat_grid_prim_path) or changed

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
