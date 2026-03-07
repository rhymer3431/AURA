from __future__ import annotations

import argparse

from locomotion.args import BOOTSTRAP_ARGS, BOOTSTRAP_PARSER, add_runtime_args

DEFAULT_DUAL_INSTRUCTION = "Navigate safely to the target and stop when complete."
DEFAULT_OBJECT_SEARCH_INSTRUCTION = "Find the bright red cube in the warehouse and stop when you reach it."
DEFAULT_INTERACTIVE_PROMPT = "nl>"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run G1 NavDP navigation through the ONNX locomotion bridge in Isaac Sim.",
        parents=[BOOTSTRAP_PARSER],
    )
    add_runtime_args(parser)
    parser.set_defaults(env_url="/Isaac/Environments/Simple_Warehouse/warehouse.usd")

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
    parser.add_argument("--strict-d455", dest="strict_d455", action="store_true")
    parser.add_argument("--force-runtime-camera", dest="force_runtime_camera", action="store_true")
    parser.add_argument("--use-trajectory-z", dest="use_trajectory_z", action="store_true")
    parser.add_argument("--image-width", dest="image_width", type=int, default=640)
    parser.add_argument("--image-height", dest="image_height", type=int, default=640)
    parser.add_argument("--depth-max-m", dest="depth_max_m", type=float, default=5.0)
    parser.add_argument("--timeout-sec", dest="timeout_sec", type=float, default=5.0)
    parser.add_argument("--reset-timeout-sec", dest="reset_timeout_sec", type=float, default=15.0)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--stop-threshold", dest="stop_threshold", type=float, default=-3.0)
    parser.add_argument("--memory-db-path", dest="memory_db_path", type=str, default="state/memory/memory.sqlite")
    parser.add_argument("--detector-engine-path", dest="detector_engine_path", type=str, default="")
    parser.add_argument("--navdp-backend", dest="navdp_backend", type=str, choices=("auto", "policy", "heuristic"), default="auto")
    parser.add_argument("--navdp-checkpoint", dest="navdp_checkpoint", type=str, default="")
    parser.add_argument("--navdp-device", dest="navdp_device", type=str, default="cpu")
    parser.add_argument("--navdp-amp", dest="navdp_amp", action="store_true")
    parser.add_argument("--navdp-amp-dtype", dest="navdp_amp_dtype", type=str, default="float16")
    parser.add_argument("--navdp-tf32", dest="navdp_tf32", action="store_true")
    parser.add_argument("--plan-wait-timeout-sec", dest="plan_wait_timeout_sec", type=float, default=0.5)
    parser.add_argument("--startup-updates", dest="startup_updates", type=int, default=20)
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=30)
    parser.add_argument("--interactive-prompt", dest="interactive_prompt", type=str, default=DEFAULT_INTERACTIVE_PROMPT)
    parser.add_argument("--interactive-idle-log-interval", dest="interactive_idle_log_interval", type=int, default=120)

    parser.add_argument("--cmd-max-vx", dest="cmd_max_vx", type=float, default=0.5)
    parser.add_argument("--cmd-max-vy", dest="cmd_max_vy", type=float, default=0.3)
    parser.add_argument("--cmd-max-wz", dest="cmd_max_wz", type=float, default=0.8)
    parser.add_argument("--lookahead-distance-m", dest="lookahead_distance_m", type=float, default=0.6)
    parser.add_argument("--heading-slowdown-rad", dest="heading_slowdown_rad", type=float, default=0.6)
    parser.add_argument("--traj-stale-timeout-sec", dest="traj_stale_timeout_sec", type=float, default=1.5)
    parser.add_argument("--cmd-accel-limit", dest="cmd_accel_limit", type=float, default=1.0)
    parser.add_argument("--cmd-yaw-accel-limit", dest="cmd_yaw_accel_limit", type=float, default=1.5)
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

    if planner_mode == "pointgoal":
        if args.goal_x is None or args.goal_y is None:
            raise ValueError("--goal-x and --goal-y are required in planner-mode=pointgoal")
        if bool(args.spawn_demo_object):
            raise ValueError("--spawn-demo-object requires --planner-mode dual")
    elif planner_mode == "dual":
        if str(args.instruction).strip() == "":
            raise ValueError("--instruction must be non-empty in planner-mode=dual")
    else:
        if bool(args.spawn_demo_object):
            raise ValueError("--spawn-demo-object requires --planner-mode dual")
        if str(args.interactive_prompt).strip() == "":
            raise ValueError("--interactive-prompt must be non-empty in planner-mode=interactive")


__all__ = [
    "BOOTSTRAP_ARGS",
    "BOOTSTRAP_PARSER",
    "DEFAULT_DUAL_INSTRUCTION",
    "DEFAULT_INTERACTIVE_PROMPT",
    "DEFAULT_OBJECT_SEARCH_INSTRUCTION",
    "apply_demo_defaults",
    "build_arg_parser",
    "validate_args",
]
