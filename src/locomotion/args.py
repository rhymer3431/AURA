"""CLI argument parsing for standalone G1 ONNX runtimes."""

from __future__ import annotations

import argparse

from .constants import ACTION_SCALE, DEFAULT_DECIMATION, DEFAULT_PHYSICS_DT


BOOTSTRAP_PARSER = argparse.ArgumentParser(add_help=False)
BOOTSTRAP_PARSER.add_argument("--headless", action="store_true", help="Run Isaac Sim without a GUI window.")
BOOTSTRAP_ARGS, _ = BOOTSTRAP_PARSER.parse_known_args()


def add_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help=(
            "Path to a locomotion policy file. Supports ONNX (`.onnx`) and TensorRT engines (`.engine`). "
            "Defaults to the exported ONNX policy in this repo root, with fallbacks to known export locations."
        ),
    )
    parser.add_argument(
        "--action_scale",
        "--action-scale",
        dest="action_scale",
        type=float,
        default=ACTION_SCALE,
        help="Scale factor applied to the policy output before joint targets are sent to the G1 articulation.",
    )
    parser.add_argument(
        "--robot_usd",
        "--robot-usd",
        "--usd-path",
        dest="robot_usd",
        type=str,
        default=None,
        help="Path to G1 USD. Defaults to src/locomotion/g1/g1_d455.usd with fallback to g1_play/g1/g1_d455.usd.",
    )
    parser.add_argument(
        "--scene_usd",
        "--scene-usd",
        dest="scene_usd",
        type=str,
        default=None,
        help="Optional absolute path to a USD/USDA environment file to reference into the stage.",
    )
    parser.add_argument(
        "--env_url",
        "--env-url",
        dest="env_url",
        type=str,
        default="/Isaac/Environments/Grid/default_environment.usd",
        help="Isaac Sim asset-relative environment path used when --scene_usd is omitted.",
    )
    parser.add_argument(
        "--scene_prim_path",
        "--scene-prim-path",
        dest="scene_prim_path",
        type=str,
        default="/World/Environment",
        help="Prim path where the environment USD is instanced.",
    )
    parser.add_argument(
        "--scene_translate",
        "--scene-translate",
        dest="scene_translate",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="World translation applied when placing the environment.",
    )
    parser.add_argument(
        "--robot_prim_path",
        "--robot-prim-path",
        dest="robot_prim_path",
        type=str,
        default="/World/G1",
        help="Prim path where the G1 robot is spawned.",
    )
    parser.add_argument(
        "--robot_position",
        "--robot-position",
        dest="robot_position",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.8),
        metavar=("X", "Y", "Z"),
        help="Initial robot base position.",
    )
    parser.add_argument(
        "--cmd_vel_timeout",
        "--cmd-vel-timeout",
        dest="cmd_vel_timeout",
        type=float,
        default=0.0,
        help="Seconds to keep the last cmd_vel command before forcing zero. Set <= 0 to keep the last command.",
    )
    parser.add_argument(
        "--onnx_device",
        "--onnx-device",
        dest="onnx_device",
        type=str,
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device preference. TensorRT engines require CUDA; ONNX policies use ONNX Runtime providers.",
    )
    parser.add_argument(
        "--physics_dt",
        "--physics-dt",
        dest="physics_dt",
        type=float,
        default=DEFAULT_PHYSICS_DT,
        help="Physics timestep in seconds. Default matches the Isaac Lab training env.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=DEFAULT_DECIMATION,
        help="Policy inference period in physics steps. Default matches the Isaac Lab training env.",
    )
    parser.add_argument(
        "--rendering_dt",
        "--rendering-dt",
        dest="rendering_dt",
        type=float,
        default=0.0,
        help="Render timestep in seconds. Defaults to decimation * physics_dt when set to 0.",
    )
    parser.add_argument(
        "--max_steps",
        "--max-steps",
        dest="max_steps",
        type=int,
        default=0,
        help="Maximum number of simulation steps before exit. 0 means run until closed.",
    )
    return parser


def build_arg_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
        or "Play a deployed G1 locomotion policy in Isaac Sim standalone mode with console cmd_vel control.",
        parents=[BOOTSTRAP_PARSER],
    )
    return add_runtime_args(parser)
