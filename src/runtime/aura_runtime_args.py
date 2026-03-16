from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from locomotion.args import BOOTSTRAP_ARGS, BOOTSTRAP_PARSER, add_runtime_args

DEFAULT_DUAL_INSTRUCTION = "Navigate safely to the target and stop when complete."
DEFAULT_OBJECT_SEARCH_INSTRUCTION = "Find the bright red cube in the warehouse and stop when you reach it."
DEFAULT_INTERACTIVE_PROMPT = "nl>"
DEFAULT_VIEWER_CONTROL_ENDPOINT = "tcp://127.0.0.1:5580"
DEFAULT_VIEWER_TELEMETRY_ENDPOINT = "tcp://127.0.0.1:5581"
DEFAULT_VIEWER_SHM_NAME = "g1_view_frames"
DEFAULT_VIEWER_SHM_SLOT_SIZE = 8 * 1024 * 1024
DEFAULT_VIEWER_SHM_CAPACITY = 8
DEFAULT_NATIVE_VIEWER = "off"


def add_subgoal_executor_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--strict-d455", dest="strict_d455", action="store_true")
    parser.add_argument("--force-runtime-camera", dest="force_runtime_camera", action="store_true")
    parser.add_argument("--use-trajectory-z", dest="use_trajectory_z", action="store_true")
    parser.add_argument("--image-width", dest="image_width", type=int, default=448)
    parser.add_argument("--image-height", dest="image_height", type=int, default=448)
    parser.add_argument("--depth-max-m", dest="depth_max_m", type=float, default=5.0)
    parser.add_argument("--memory-db-path", dest="memory_db_path", type=str, default="state/memory/memory.sqlite")
    memory_store_group = parser.add_mutually_exclusive_group()
    memory_store_group.add_argument("--memory-store", dest="memory_store", action="store_true")
    memory_store_group.add_argument("--no-memory-store", dest="memory_store", action="store_false")
    parser.set_defaults(memory_store=True)
    parser.add_argument("--skip-detection", dest="skip_detection", action="store_true")
    parser.add_argument("--detector-model-path", dest="detector_model_path", type=str, default="")
    parser.add_argument("--detector-device", dest="detector_device", type=str, default="")
    parser.add_argument("--navdp-backend", dest="navdp_backend", type=str, choices=("auto", "policy", "heuristic"), default="auto")
    parser.add_argument("--navdp-checkpoint", dest="navdp_checkpoint", type=str, default="")
    parser.add_argument("--navdp-device", dest="navdp_device", type=str, default="cpu")
    parser.add_argument("--navdp-amp", dest="navdp_amp", action="store_true")
    parser.add_argument("--navdp-amp-dtype", dest="navdp_amp_dtype", type=str, default="float16")
    parser.add_argument("--navdp-tf32", dest="navdp_tf32", action="store_true")
    parser.add_argument("--plan-wait-timeout-sec", dest="plan_wait_timeout_sec", type=float, default=0.5)
    parser.add_argument("--startup-updates", dest="startup_updates", type=int, default=20)
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=30)
    parser.add_argument("--cmd-max-vx", dest="cmd_max_vx", type=float, default=0.5)
    parser.add_argument("--cmd-max-vy", dest="cmd_max_vy", type=float, default=0.3)
    parser.add_argument("--cmd-max-wz", dest="cmd_max_wz", type=float, default=0.8)
    parser.add_argument("--lookahead-distance-m", dest="lookahead_distance_m", type=float, default=0.6)
    parser.add_argument("--heading-slowdown-rad", dest="heading_slowdown_rad", type=float, default=0.6)
    parser.add_argument("--traj-stale-timeout-sec", dest="traj_stale_timeout_sec", type=float, default=1.5)
    parser.add_argument("--cmd-accel-limit", dest="cmd_accel_limit", type=float, default=1.0)
    parser.add_argument("--cmd-yaw-accel-limit", dest="cmd_yaw_accel_limit", type=float, default=1.5)
    parser.add_argument("--obstacle-stop-distance-m", dest="obstacle_stop_distance_m", type=float, default=0.45)
    parser.add_argument(
        "--obstacle-hold-distance-m",
        "--obstacle-turn-distance-m",
        dest="obstacle_hold_distance_m",
        type=float,
        default=0.70,
    )
    parser.add_argument("--obstacle-slow-forward-vx-mps", dest="obstacle_slow_forward_vx_mps", type=float, default=0.08)
    parser.add_argument("--obstacle-backoff-vx-mps", dest="obstacle_backoff_vx_mps", type=float, default=0.18)
    parser.add_argument("--obstacle-lateral-nudge-vy-mps", dest="obstacle_lateral_nudge_vy_mps", type=float, default=0.12)
    parser.add_argument("--obstacle-recovery-hold-sec", dest="obstacle_recovery_hold_sec", type=float, default=0.75)
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AURA runtime for G1 navigation in Isaac Sim.",
        parents=[BOOTSTRAP_PARSER],
    )
    add_runtime_args(parser)
    parser.set_defaults(env_url="/Isaac/Environments/Simple_Warehouse/warehouse.usd")
    parser.add_argument("--launch-mode", dest="launch_mode", type=str, choices=("gui", "g1_view"), default="")

    parser.add_argument("--server-url", dest="server_url", type=str, default="http://127.0.0.1:8888")
    parser.add_argument(
        "--planner-mode",
        dest="planner_mode",
        type=str,
        choices=("pointgoal", "dual", "interactive"),
        default="interactive",
    )
    parser.add_argument("--dual-server-url", dest="dual_server_url", type=str, default="http://127.0.0.1:8890")
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_DUAL_INSTRUCTION,
    )
    parser.add_argument("--goal-x", dest="goal_x", type=float, default=None)
    parser.add_argument("--goal-y", dest="goal_y", type=float, default=None)
    parser.add_argument("--goal-tolerance-m", dest="goal_tolerance_m", type=float, default=0.4)
    parser.add_argument("--exit-on-pointgoal-failure", dest="exit_on_pointgoal_failure", action="store_true")
    parser.add_argument("--no-exit-on-pointgoal-failure", dest="exit_on_pointgoal_failure", action="store_false")
    parser.set_defaults(exit_on_pointgoal_failure=True)
    parser.add_argument("--global-map-image", dest="global_map_image", type=str, default="")
    parser.add_argument("--global-map-config", dest="global_map_config", type=str, default="")
    parser.add_argument("--global-waypoint-spacing-m", dest="global_waypoint_spacing_m", type=float, default=0.75)
    parser.add_argument("--global-inflation-radius-m", dest="global_inflation_radius_m", type=float, default=0.25)
    parser.add_argument("--spawn-demo-object", dest="spawn_demo_object", action="store_true")
    parser.add_argument("--demo-object-x", dest="demo_object_x", type=float, default=2.0)
    parser.add_argument("--demo-object-y", dest="demo_object_y", type=float, default=0.0)
    parser.add_argument("--demo-object-size-m", dest="demo_object_size_m", type=float, default=0.25)
    parser.add_argument("--object-stop-radius-m", dest="object_stop_radius_m", type=float, default=0.8)
    parser.add_argument("--plan-interval-frames", dest="plan_interval_frames", type=int, default=3)
    parser.add_argument("--dual-request-gap-frames", dest="dual_request_gap_frames", type=int, default=3)
    parser.add_argument("--safety-timeout-sec", dest="safety_timeout_sec", type=float, default=20.0)
    parser.add_argument("--s1-period-sec", dest="s1_period_sec", type=float, default=0.2)
    parser.add_argument("--s2-period-sec", dest="s2_period_sec", type=float, default=1.0)
    parser.add_argument("--goal-ttl-sec", dest="goal_ttl_sec", type=float, default=3.0)
    parser.add_argument("--traj-ttl-sec", dest="traj_ttl_sec", type=float, default=1.5)
    parser.add_argument("--traj-max-stale-sec", dest="traj_max_stale_sec", type=float, default=4.0)
    parser.add_argument("--timeout-sec", dest="timeout_sec", type=float, default=5.0)
    parser.add_argument("--reset-timeout-sec", dest="reset_timeout_sec", type=float, default=15.0)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--stop-threshold", dest="stop_threshold", type=float, default=-3.0)
    parser.add_argument("--interactive-prompt", dest="interactive_prompt", type=str, default=DEFAULT_INTERACTIVE_PROMPT)
    parser.add_argument("--interactive-idle-log-interval", dest="interactive_idle_log_interval", type=int, default=120)
    parser.add_argument("--viewer-control-endpoint", dest="viewer_control_endpoint", type=str, default=DEFAULT_VIEWER_CONTROL_ENDPOINT)
    parser.add_argument("--viewer-telemetry-endpoint", dest="viewer_telemetry_endpoint", type=str, default=DEFAULT_VIEWER_TELEMETRY_ENDPOINT)
    parser.add_argument("--viewer-shm-name", dest="viewer_shm_name", type=str, default=DEFAULT_VIEWER_SHM_NAME)
    parser.add_argument("--viewer-shm-slot-size", dest="viewer_shm_slot_size", type=int, default=DEFAULT_VIEWER_SHM_SLOT_SIZE)
    parser.add_argument("--viewer-shm-capacity", dest="viewer_shm_capacity", type=int, default=DEFAULT_VIEWER_SHM_CAPACITY)
    parser.add_argument("--viewer-publish", dest="viewer_publish", action="store_true")
    parser.add_argument("--no-viewer-publish", dest="viewer_publish", action="store_false")
    parser.set_defaults(viewer_publish=False)
    parser.add_argument("--native-viewer", dest="native_viewer", type=str, choices=("off", "opencv"), default=DEFAULT_NATIVE_VIEWER)
    parser.add_argument("--show-depth", dest="show_depth", action="store_true")
    add_subgoal_executor_args(parser)
    return parser


def apply_demo_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if bool(getattr(args, "spawn_demo_object", False)) and str(args.planner_mode).lower() == "dual":
        instruction = str(getattr(args, "instruction", "")).strip()
        if instruction == "" or instruction == DEFAULT_DUAL_INSTRUCTION:
            args.instruction = DEFAULT_OBJECT_SEARCH_INSTRUCTION
    return args


def validate_args(args: argparse.Namespace) -> None:
    planner_mode = str(args.planner_mode).lower()
    if float(args.demo_object_size_m) <= 0.0:
        raise ValueError("--demo-object-size-m must be positive")
    if float(args.object_stop_radius_m) <= 0.0:
        raise ValueError("--object-stop-radius-m must be positive")
    if int(args.interactive_idle_log_interval) <= 0:
        raise ValueError("--interactive-idle-log-interval must be positive")
    if float(args.obstacle_stop_distance_m) <= 0.0:
        raise ValueError("--obstacle-stop-distance-m must be positive")
    if float(args.obstacle_hold_distance_m) <= 0.0:
        raise ValueError("--obstacle-hold-distance-m must be positive")
    if float(args.obstacle_hold_distance_m) < float(args.obstacle_stop_distance_m):
        raise ValueError("--obstacle-hold-distance-m must be greater than or equal to --obstacle-stop-distance-m")
    if float(args.obstacle_recovery_hold_sec) < 0.0:
        raise ValueError("--obstacle-recovery-hold-sec must be non-negative")
    if float(getattr(args, "global_waypoint_spacing_m", 0.75)) <= 0.0:
        raise ValueError("--global-waypoint-spacing-m must be positive")
    if float(getattr(args, "global_inflation_radius_m", 0.25)) < 0.0:
        raise ValueError("--global-inflation-radius-m must be non-negative")
    global_map_image = str(getattr(args, "global_map_image", "")).strip()
    global_map_config = str(getattr(args, "global_map_config", "")).strip()
    if global_map_config != "" and global_map_image == "":
        raise ValueError("--global-map-config requires --global-map-image")

    if planner_mode == "pointgoal":
        if args.goal_x is None or args.goal_y is None:
            raise ValueError("--goal-x and --goal-y are required in planner-mode=pointgoal")
        if bool(args.spawn_demo_object):
            raise ValueError("--spawn-demo-object requires --planner-mode dual")
        if global_map_image != "":
            if not Path(global_map_image).exists():
                raise ValueError(f"--global-map-image not found: {global_map_image}")
            if global_map_config != "" and not Path(global_map_config).exists():
                raise ValueError(f"--global-map-config not found: {global_map_config}")
    elif planner_mode == "dual":
        if str(args.instruction).strip() == "":
            raise ValueError("--instruction must be non-empty in planner-mode=dual")
        if str(getattr(args, "global_map_image", "")).strip() != "":
            raise ValueError("--global-map-image requires --planner-mode pointgoal")
    else:
        if bool(args.spawn_demo_object):
            raise ValueError("--spawn-demo-object requires --planner-mode dual")
        if str(args.interactive_prompt).strip() == "":
            raise ValueError("--interactive-prompt must be non-empty in planner-mode=interactive")
        if str(getattr(args, "global_map_image", "")).strip() != "":
            raise ValueError("--global-map-image requires --planner-mode pointgoal")
    launch_mode = str(getattr(args, "launch_mode", "")).strip().lower()
    if (
        str(getattr(args, "native_viewer", DEFAULT_NATIVE_VIEWER)).strip().lower() == "opencv"
        and not bool(getattr(args, "viewer_publish", False))
        and launch_mode != "g1_view"
    ):
        raise ValueError("--native-viewer opencv requires --viewer-publish")


def resolve_launch_mode(args: argparse.Namespace) -> Literal["gui", "g1_view", "headless"]:
    raw_mode = str(getattr(args, "launch_mode", "")).strip().lower()
    if raw_mode == "gui":
        return "gui"
    if raw_mode == "g1_view":
        return "g1_view"
    return "headless" if bool(getattr(args, "headless", False)) else "gui"


def apply_launch_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    resolved_mode = resolve_launch_mode(args)
    args.resolved_launch_mode = resolved_mode
    args.headless = resolved_mode != "gui"
    native_viewer = str(getattr(args, "native_viewer", DEFAULT_NATIVE_VIEWER)).strip().lower() or DEFAULT_NATIVE_VIEWER
    viewer_publish = bool(getattr(args, "viewer_publish", False))
    if resolved_mode == "g1_view":
        viewer_publish = True
        native_viewer = "opencv"
    args.viewer_publish = viewer_publish
    args.native_viewer = native_viewer
    return args


__all__ = [
    "BOOTSTRAP_ARGS",
    "BOOTSTRAP_PARSER",
    "DEFAULT_DUAL_INSTRUCTION",
    "DEFAULT_INTERACTIVE_PROMPT",
    "DEFAULT_OBJECT_SEARCH_INSTRUCTION",
    "DEFAULT_VIEWER_CONTROL_ENDPOINT",
    "DEFAULT_NATIVE_VIEWER",
    "DEFAULT_VIEWER_TELEMETRY_ENDPOINT",
    "DEFAULT_VIEWER_SHM_NAME",
    "DEFAULT_VIEWER_SHM_SLOT_SIZE",
    "DEFAULT_VIEWER_SHM_CAPACITY",
    "add_subgoal_executor_args",
    "apply_launch_mode_defaults",
    "apply_demo_defaults",
    "build_arg_parser",
    "resolve_launch_mode",
    "validate_args",
]
