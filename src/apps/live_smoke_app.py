from __future__ import annotations

import argparse
from pathlib import Path

from locomotion.args import BOOTSTRAP_PARSER, add_runtime_args
from runtime.bootstrap_profiles import list_profile_names
from runtime.isaac_launch_modes import (
    LAUNCH_MODE_AUTO,
    LAUNCH_MODE_EDITOR_ASSISTED,
    LAUNCH_MODE_ATTACH,
    LAUNCH_MODE_EXTENSION,
    LAUNCH_MODE_STANDALONE,
)
from runtime.live_smoke_runner import LiveSmokeRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac live smoke and bootstrap diagnostics.", parents=[BOOTSTRAP_PARSER])
    add_runtime_args(parser)
    parser.set_defaults(env_url="/Isaac/Environments/Simple_Warehouse/warehouse.usd")
    parser.add_argument("--mode", choices=("preflight", "smoke"), default="smoke")
    parser.add_argument(
        "--launch-mode",
        choices=(LAUNCH_MODE_AUTO, LAUNCH_MODE_STANDALONE, LAUNCH_MODE_EDITOR_ASSISTED, LAUNCH_MODE_ATTACH, LAUNCH_MODE_EXTENSION),
        default=LAUNCH_MODE_STANDALONE,
    )
    parser.add_argument("--bootstrap-profile", choices=list_profile_names(), default="auto")
    parser.add_argument("--smoke-target-tier", choices=("sensor", "pipeline", "memory", "full"), default="sensor")
    parser.add_argument("--frame-source", choices=("live",), default="live")
    parser.add_argument("--diagnostics-path", type=str, default="tmp/process_logs/live_smoke/diagnostics.json")
    parser.add_argument("--artifacts-dir", type=str, default="tmp/process_logs/live_smoke")
    parser.add_argument("--assets-root", type=str, default="")
    parser.add_argument("--d455-asset-path", type=str, default="")
    parser.add_argument("--experience-path", type=str, default="")
    parser.add_argument("--d455-prim-path", type=str, default="/World/realsense_d455")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=640)
    parser.add_argument("--depth-max-m", type=float, default=5.0)
    parser.add_argument("--detector-model-path", type=str, default="")
    parser.add_argument("--detector-device", type=str, default="")
    parser.add_argument("--startup-updates", type=int, default=4)
    parser.add_argument("--force-runtime-camera", action="store_true")
    parser.add_argument("--render-warmup-updates", type=int, default=-1)
    parser.add_argument("--physics-warmup-steps", type=int, default=-1)
    parser.add_argument("--stage-settle-updates", type=int, default=-1)
    parser.add_argument("--sensor-init-retries", type=int, default=-1)
    parser.add_argument("--sensor-init-retry-updates", type=int, default=-1)
    parser.add_argument("--app-bootstrap-timeout-sec", type=float, default=90.0)
    parser.add_argument("--stage-ready-timeout-sec", type=float, default=45.0)
    parser.add_argument("--sensor-init-timeout-sec", type=float, default=45.0)
    parser.add_argument("--first-frame-timeout-sec", type=float, default=20.0)
    args = parser.parse_args(argv)
    setattr(args, "_argv", list(argv or []))
    args.artifacts_dir = str(Path(args.artifacts_dir))
    args.diagnostics_path = str(Path(args.diagnostics_path))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = LiveSmokeRunner(args)
    if args.mode == "preflight":
        return runner.run_preflight()
    return runner.run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
