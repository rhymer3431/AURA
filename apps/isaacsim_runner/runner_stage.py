from __future__ import annotations

"""Deprecated: use apps.isaacsim_runner.config.stage and apps.isaacsim_runner.stage.* instead."""

from apps.isaacsim_runner.config.stage import (
    StageLayoutConfig,
    StageReferenceSpec,
    _sanitize_stage_token,
    _split_reference_spec,
    parse_stage_reference_specs,
)
from apps.isaacsim_runner.stage.layout import (
    _add_flat_grid,
    _add_reference_to_stage_prim,
    _apply_stage_references,
    _ensure_stage_default_prim,
    _ensure_stage_hierarchy,
    _ensure_stage_key_light,
    _ensure_stage_physics_scene,
    _move_prim_if_needed,
    apply_stage_layout,
    ensure_world_environment,
)
from apps.isaacsim_runner.stage.prims import (
    ensure_parent_xforms,
    get_translate_op,
    normalize_prim_path,
    quat_to_rpy,
    read_base_pose,
)
from apps.isaacsim_runner.stage.robot import (
    _count_rigid_bodies_under,
    ensure_robot_dynamic_flags,
    find_camera_prim_path,
    find_motion_root_prim,
    find_robot_prim_path,
    rebase_world_anchor_articulation_root,
    resolve_robot_placement_prim,
    set_robot_start_height,
)

_parse_stage_reference_specs = parse_stage_reference_specs
_normalize_prim_path = normalize_prim_path
_get_translate_op = get_translate_op
_quat_to_rpy = quat_to_rpy
_read_base_pose = read_base_pose
_ensure_parent_xforms = ensure_parent_xforms
_ensure_world_environment = ensure_world_environment
_apply_stage_layout = apply_stage_layout
_add_flat_grid_environment = _add_flat_grid
_find_robot_prim_path = find_robot_prim_path
_resolve_robot_placement_prim = resolve_robot_placement_prim
_rebase_world_anchor_articulation_root = rebase_world_anchor_articulation_root
_ensure_robot_dynamic_flags = ensure_robot_dynamic_flags
_find_motion_root_prim = find_motion_root_prim
_set_robot_start_height = set_robot_start_height
_find_camera_prim_path = find_camera_prim_path
