"""CLI argument parsing for the standalone G1 locomotion runner."""

from __future__ import annotations

import argparse

DEFAULT_PHYSICS_DT = 1.0 / 200.0
DEFAULT_DECIMATION = 4


BOOTSTRAP_PARSER = argparse.ArgumentParser(add_help=False)
BOOTSTRAP_PARSER.add_argument("--headless", action="store_true", help="Run Isaac Sim without a GUI window.")
BOOTSTRAP_ARGS, _ = BOOTSTRAP_PARSER.parse_known_args()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a deployed G1 locomotion policy in Isaac Sim standalone mode.",
        parents=[BOOTSTRAP_PARSER],
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help=(
            "Path to a locomotion policy file. Supports ONNX (`.onnx`) and TensorRT engines (`.engine`). "
            "Defaults to policy_fp16_b1.engine in this repo, with fallback to compatible ONNX exports."
        ),
    )
    parser.add_argument(
        "--robot_usd",
        type=str,
        default=None,
        help="Path to G1 USD. Defaults to robots/g1/g1_d455.usd in this repo.",
    )
    parser.add_argument(
        "--scene_usd",
        type=str,
        default=None,
        help="Optional absolute path to a USD/USDA environment file to reference into the stage.",
    )
    parser.add_argument(
        "--env_url",
        type=str,
        default="/Isaac/Environments/Grid/default_environment.usd",
        help="Isaac Sim asset-relative environment path used when --scene_usd is omitted.",
    )
    parser.add_argument(
        "--scene_prim_path",
        type=str,
        default="/World/Environment",
        help="Prim path where the environment USD is instanced.",
    )
    parser.add_argument(
        "--scene_translate",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="World translation applied when placing the environment.",
    )
    parser.add_argument(
        "--robot_prim_path",
        type=str,
        default="/World/G1",
        help="Prim path where the G1 robot is spawned.",
    )
    parser.add_argument(
        "--robot_position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Initial robot base position. Defaults to the training config when provided, else (0.0, 0.0, 0.8).",
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        choices=("cmd_vel", "keyboard", "internvla_navdp"),
        default="cmd_vel",
        help=(
            "Operator input source. 'cmd_vel' reads vx vy wz from the console, "
            "'keyboard' reads GUI keyboard events, and 'internvla_navdp' follows local trajectories "
            "provided by the standalone navigation system."
        ),
    )
    parser.add_argument(
        "--cmd_vel_timeout",
        type=float,
        default=0.0,
        help="Seconds to keep the last cmd_vel command before forcing zero. Set <= 0 to keep the last command.",
    )
    parser.add_argument(
        "--lin_speed",
        type=float,
        default=0.8,
        help="Keyboard forward/backward speed for W/S or UP/DOWN when --control_mode keyboard is used.",
    )
    parser.add_argument(
        "--lat_speed",
        type=float,
        default=0.4,
        help="Keyboard lateral speed for Q/E when --control_mode keyboard is used.",
    )
    parser.add_argument(
        "--yaw_speed",
        type=float,
        default=1.0,
        help="Keyboard yaw speed for A/D or LEFT/RIGHT when --control_mode keyboard is used.",
    )
    parser.add_argument(
        "--require_keyboard_focus",
        action="store_true",
        help="Fail fast if GUI keyboard input is unavailable instead of continuing with zero commands.",
    )
    parser.add_argument(
        "--onnx_device",
        type=str,
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device preference. TensorRT engines require CUDA; ONNX policies use ONNX Runtime providers.",
    )
    parser.add_argument(
        "--physics_dt",
        type=float,
        default=None,
        help=f"Physics timestep in seconds. Defaults to the training config when provided, else {DEFAULT_PHYSICS_DT}.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=None,
        help=f"Policy inference period in physics steps. Defaults to the training config when provided, else {DEFAULT_DECIMATION}.",
    )
    parser.add_argument(
        "--rendering_dt",
        type=float,
        default=0.0,
        help="Render timestep in seconds. Defaults to decimation * physics_dt when set to 0.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum number of simulation steps before exit. 0 means run until closed.",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=None,
        help="Optional policy action scale override. Defaults are inferred from the policy engine observation shape.",
    )
    parser.add_argument(
        "--height_scan_size",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional height scan footprint override in meters. Defaults are inferred from the policy engine observation shape.",
    )
    parser.add_argument(
        "--height_scan_resolution",
        type=float,
        default=None,
        help="Optional height scan grid resolution override in meters.",
    )
    parser.add_argument(
        "--height_scan_offset",
        type=float,
        default=None,
        help="Optional height scan vertical offset override. Defaults to the training-time offset.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Optional exported training config directory. When set, runtime defaults are loaded from env.yaml inside it.",
    )
    parser.add_argument(
        "--navigation_url",
        type=str,
        default="http://127.0.0.1:17882",
        help="Base URL of the standalone navigation system.",
    )
    parser.add_argument(
        "--navigation_timeout",
        type=float,
        default=5.0,
        help="HTTP timeout in seconds for calls to the navigation system.",
    )
    parser.add_argument(
        "--navigation_trajectory_timeout",
        type=float,
        default=2.0,
        help="Maximum age in seconds for following the latest navigation-system trajectory before holding zero commands.",
    )
    parser.add_argument(
        "--navigation_update_hz",
        type=float,
        default=3.0,
        help="Maximum rate used when publishing observations to the navigation system.",
    )
    parser.add_argument(
        "--runtime_control_api_host",
        type=str,
        default="127.0.0.1",
        help="Bind address for the runtime control HTTP API in internvla_navdp mode.",
    )
    parser.add_argument(
        "--runtime_control_api_port",
        type=int,
        default=8892,
        help="Port for the runtime control HTTP API. Set to 0 to disable it.",
    )
    parser.add_argument(
        "--camera_prim_path",
        type=str,
        default=None,
        help=(
            "Optional camera rig or camera prim path. Defaults to <robot_prim_path>/head_link/Realsense "
            "and falls back to a direct camera prim when that path is itself a Camera."
        ),
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=448,
        help="Navigation camera width in pixels.",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=448,
        help="Navigation camera height in pixels.",
    )
    parser.add_argument(
        "--camera_pos",
        type=float,
        nargs=3,
        default=(0.2, 0.0, 1.1),
        metavar=("X", "Y", "Z"),
        help="Local camera translation relative to the robot root prim.",
    )
    parser.add_argument(
        "--camera_quat",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
        help="Local camera orientation as a scalar-first quaternion.",
    )
    parser.add_argument(
        "--camera_pitch_deg",
        type=float,
        default=0.0,
        help="Extra local camera pitch in degrees. Positive pitches the camera upward.",
    )
    parser.add_argument(
        "--camera_pitch_min_deg",
        type=float,
        default=-45.0,
        help="Lower clamp for runtime camera pitch control in degrees.",
    )
    parser.add_argument(
        "--camera_pitch_max_deg",
        type=float,
        default=45.0,
        help="Upper clamp for runtime camera pitch control in degrees.",
    )
    parser.add_argument(
        "--camera_near",
        type=float,
        default=0.05,
        help="Navigation camera near clipping distance in meters.",
    )
    parser.add_argument(
        "--camera_far",
        type=float,
        default=20.0,
        help="Navigation camera far clipping distance in meters.",
    )
    parser.add_argument(
        "--camera_api_host",
        type=str,
        default="127.0.0.1",
        help="Bind address for the camera pitch HTTP API.",
    )
    parser.add_argument(
        "--camera_api_port",
        type=int,
        default=0,
        help="Port for the camera pitch HTTP API. Set to 0 to disable the API server.",
    )
    viewer_publish_group = parser.add_mutually_exclusive_group()
    viewer_publish_group.add_argument(
        "--viewer-publish",
        dest="viewer_publish",
        action="store_true",
        help="Publish Isaac POV frames onto the dashboard viewer transport.",
    )
    viewer_publish_group.add_argument(
        "--no-viewer-publish",
        dest="viewer_publish",
        action="store_false",
        help="Disable Isaac POV frame publication onto the dashboard viewer transport.",
    )
    parser.set_defaults(viewer_publish=True)
    parser.add_argument(
        "--lookahead_distance",
        type=float,
        default=0.75,
        help="Follower lookahead distance in meters for tracking NavDP trajectories.",
    )
    parser.add_argument(
        "--vx_max",
        type=float,
        default=0.5,
        help="Maximum forward velocity command sent to the locomotion policy.",
    )
    parser.add_argument(
        "--vy_max",
        type=float,
        default=0.3,
        help="Maximum lateral velocity command sent to the locomotion policy.",
    )
    parser.add_argument(
        "--wz_max",
        type=float,
        default=1.2,
        help="Maximum yaw-rate command sent to the locomotion policy.",
    )
    parser.add_argument(
        "--cmd_smoothing_tau",
        type=float,
        default=0.25,
        help="First-order smoothing constant for NavDP follower output in seconds.",
    )
    return parser
