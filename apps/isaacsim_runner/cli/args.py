from __future__ import annotations

import argparse

from apps.isaacsim_runner.config.base import default_usd_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isaac Sim 4.2 runner for g1_d455.usd on Windows."
    )
    parser.add_argument("--usd", type=str, default=str(default_usd_path()))
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
    parser.add_argument(
        "--disable-flat-grid",
        action="store_true",
        help="Disable default Isaac flat grid bootstrap.",
    )
    parser.add_argument(
        "--flat-grid-prim",
        type=str,
        default="",
        help="Override flat grid target prim path (default: <stage-environment-prim>/FlatGrid).",
    )
    parser.add_argument("--stage-world-prim", type=str, default="/World")
    parser.add_argument("--stage-environment-prim", type=str, default="/World/Environment")
    parser.add_argument("--stage-robots-prim", type=str, default="/World/Robots")
    parser.add_argument("--stage-object-root-prim", type=str, default="/World/Environment/Objects")
    parser.add_argument("--stage-physics-scene-prim", type=str, default="/World/PhysicsScene")
    parser.add_argument("--stage-key-light-prim", type=str, default="/World/Environment/KeyLight")
    parser.add_argument("--stage-key-light-intensity", type=float, default=500.0)
    parser.add_argument("--stage-key-light-angle", type=float, default=0.53)
    parser.add_argument(
        "--environment-ref",
        action="append",
        default=[],
        metavar="USD[@/Prim/Path]",
        help="Add environment USD reference. Repeatable. If prim omitted, auto-place under stage environment prim.",
    )
    parser.add_argument(
        "--object-ref",
        action="append",
        default=[],
        metavar="USD[@/Prim/Path]",
        help="Add object USD reference. Repeatable. If prim omitted, auto-place under stage object root prim.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode even if Isaac Sim python modules are available.",
    )
    return parser
