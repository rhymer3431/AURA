from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from telemetry_runtime import JsonlTelemetryLogger, now_perf
except Exception:  # pragma: no cover - optional telemetry dependency
    JsonlTelemetryLogger = None  # type: ignore

    def now_perf() -> float:
        return time.perf_counter()

from apps.isaacsim_runner.bridges.mock import MockRos2Publisher, run_mock_loop
from apps.isaacsim_runner.bridges.navigate import NavigateCommandBridge
from apps.isaacsim_runner.bridges.omnigraph import setup_camera_graph, setup_joint_and_tf_graph
from apps.isaacsim_runner.cli.args import build_argument_parser
from apps.isaacsim_runner.config.base import (
    DEFAULT_G1_GROUND_CLEARANCE_Z,
    DEFAULT_G1_START_Z,
    NAV_CMD_DEADBAND,
    configure_logging,
    default_usd_path,
    prepare_internal_ros2_environment,
)
from apps.isaacsim_runner.config.stage import StageLayoutConfig, StageReferenceSpec, parse_stage_reference_specs
from apps.isaacsim_runner.control.g1 import (
    ARM_JOINTS,
    LEG_JOINTS,
    PD_JOINT_ORDER,
    PD_KD,
    PD_KD_BY_NAME,
    PD_KP,
    PD_KP_BY_NAME,
    WAIST_JOINTS,
    _log_gain_summary,
    apply_pd_gains,
    log_robot_dof_snapshot,
)
from apps.isaacsim_runner.runtime.orchestrator import _run_native_isaac as _run_native_isaac_impl
from apps.isaacsim_runner.stage.layout import _add_flat_grid as _add_flat_grid_environment
from apps.isaacsim_runner.stage.layout import apply_stage_layout, ensure_world_environment
from apps.isaacsim_runner.stage.prims import get_translate_op, quat_to_rpy, read_base_pose
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

_setup_ros2_camera_graph = setup_camera_graph
_setup_ros2_joint_and_tf_graph = setup_joint_and_tf_graph
_run_mock_loop = run_mock_loop

_configure_logging = configure_logging
_default_usd_path = default_usd_path
_prepare_internal_ros2_environment = prepare_internal_ros2_environment

G1_ARM_JOINTS = ARM_JOINTS
G1_LEG_JOINTS = LEG_JOINTS
G1_PD_JOINT_ORDER = PD_JOINT_ORDER
G1_PD_KD = PD_KD
G1_PD_KD_BY_NAME = PD_KD_BY_NAME
G1_PD_KP = PD_KP
G1_PD_KP_BY_NAME = PD_KP_BY_NAME
G1_WAIST_JOINTS = WAIST_JOINTS

_apply_g1_pd_gains = apply_pd_gains
_log_robot_dof_snapshot = log_robot_dof_snapshot

_apply_stage_layout = apply_stage_layout
_ensure_world_environment = ensure_world_environment
_find_camera_prim_path = find_camera_prim_path
_find_motion_root_prim = find_motion_root_prim
_find_robot_prim_path = find_robot_prim_path
_get_translate_op = get_translate_op
_parse_stage_reference_specs = parse_stage_reference_specs
_quat_to_rpy = quat_to_rpy
_read_base_pose = read_base_pose
_rebase_world_anchor_articulation_root = rebase_world_anchor_articulation_root
_resolve_robot_placement_prim = resolve_robot_placement_prim
_set_robot_start_height = set_robot_start_height
_ensure_robot_dynamic_flags = ensure_robot_dynamic_flags


def _run_native_isaac(args: argparse.Namespace) -> None:
    _run_native_isaac_impl(args, jsonl_logger_cls=JsonlTelemetryLogger, perf_now=now_perf)


def _parse_args() -> argparse.Namespace:
    parser = build_argument_parser()
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    usd_path = Path(args.usd).resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    logging.info("Using USD: %s", usd_path)

    if args.mock:
        _run_mock_loop(args)
        return
    _run_native_isaac(args)


if __name__ == "__main__":
    main()
