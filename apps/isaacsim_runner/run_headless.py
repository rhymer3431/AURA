from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from telemetry_runtime import JsonlTelemetryLogger, now_perf
except Exception:  # pragma: no cover - optional telemetry dependency
    JsonlTelemetryLogger = None  # type: ignore

    def now_perf() -> float:
        return time.perf_counter()

from apps.isaacsim_runner.runner_bridges import (
    MockRos2Publisher,
    NavigateCommandBridge,
    _run_mock_loop,
    _setup_ros2_camera_graph,
    _setup_ros2_joint_and_tf_graph,
)
from apps.isaacsim_runner.runner_config import (
    DEFAULT_G1_GROUND_CLEARANCE_Z,
    DEFAULT_G1_START_Z,
    NAV_CMD_DEADBAND,
    _configure_logging,
    _default_usd_path,
    _prepare_internal_ros2_environment,
)
from apps.isaacsim_runner.runner_control import (
    G1_ARM_JOINTS,
    G1_LEG_JOINTS,
    G1_PD_JOINT_ORDER,
    G1_PD_KD,
    G1_PD_KD_BY_NAME,
    G1_PD_KP,
    G1_PD_KP_BY_NAME,
    G1_WAIST_JOINTS,
    _apply_g1_pd_gains,
    _log_gain_summary,
    _log_robot_dof_snapshot,
)
from apps.isaacsim_runner.runner_execution import _run_native_isaac as _run_native_isaac_impl
from apps.isaacsim_runner.runner_stage import (
    _add_flat_grid_environment,
    _count_rigid_bodies_under,
    _ensure_robot_dynamic_flags,
    _ensure_world_environment,
    _find_camera_prim_path,
    _find_motion_root_prim,
    _find_robot_prim_path,
    _get_translate_op,
    _quat_to_rpy,
    _read_base_pose,
    _rebase_world_anchor_articulation_root,
    _resolve_robot_placement_prim,
    _set_robot_start_height,
)


def _run_native_isaac(args: argparse.Namespace) -> None:
    _run_native_isaac_impl(args, jsonl_logger_cls=JsonlTelemetryLogger, perf_now=now_perf)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isaac Sim 4.2 runner for g1_d455.usd on Windows."
    )
    parser.add_argument("--usd", type=str, default=str(_default_usd_path()))
    parser.add_argument("--namespace", type=str, default="g1")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--publish-imu", action="store_true")
    parser.add_argument("--publish-compressed-color", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Run native Isaac Sim with GUI window.")
    parser.add_argument(
        "--enable-camera-bridge-in-gui",
        action="store_true",
        help="Enable ROS camera graph in GUI mode (may impact viewport interactivity on some systems).",
    )
    parser.add_argument(
        "--enable-navigate-bridge",
        action="store_true",
        help="Enable /<namespace>/cmd/navigate root transform bridge (disabled by default for physics safety).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode even if Isaac Sim python modules are available.",
    )
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
