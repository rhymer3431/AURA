"""Backend entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiohttp import web

from backend.app import create_app
from backend.webrtc import WebRTCServiceConfig


REPO_ROOT = Path(__file__).resolve().parents[3]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AURA backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18095)
    parser.add_argument("--dev-origin", default="http://127.0.0.1:5173")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:18095")
    parser.add_argument(
        "--runtime-url",
        "--runtime-supervisor-url",
        dest="runtime_url",
        default="",
        help="Optional external runtime URL. Leave unset to let the backend own runtime lifecycle locally.",
    )
    parser.add_argument("--inference-system-url", default="http://127.0.0.1:15880")
    parser.add_argument("--planner-system-url", default="http://127.0.0.1:17881")
    parser.add_argument("--navigation-system-url", default="http://127.0.0.1:17882")
    parser.add_argument("--control-runtime-url", default="http://127.0.0.1:8892")
    parser.add_argument("--webrtc-proxy-base", default="")
    parser.add_argument("--webrtc-rgb-fps", type=float, default=30.0)
    parser.add_argument("--webrtc-depth-fps", type=float, default=15.0)
    parser.add_argument("--webrtc-telemetry-hz", type=float, default=15.0)
    parser.add_argument("--webrtc-poll-interval-ms", type=int, default=10)
    parser.add_argument("--webrtc-enable-depth-track", dest="webrtc_enable_depth_track", action="store_true")
    parser.add_argument("--webrtc-disable-depth-track", dest="webrtc_enable_depth_track", action="store_false")
    parser.set_defaults(webrtc_enable_depth_track=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    app = create_app(
        root_dir=str(REPO_ROOT),
        api_base_url=str(args.api_base_url).rstrip("/"),
        dev_origin=str(args.dev_origin),
        runtime_url=str(args.runtime_url).rstrip("/"),
        inference_system_url=str(args.inference_system_url).rstrip("/"),
        planner_system_url=str(args.planner_system_url).rstrip("/"),
        navigation_system_url=str(args.navigation_system_url).rstrip("/"),
        control_runtime_url=str(args.control_runtime_url).rstrip("/"),
        webrtc_proxy_base=str(args.webrtc_proxy_base).rstrip("/"),
        webrtc_config=WebRTCServiceConfig(
            enable_depth_track=bool(args.webrtc_enable_depth_track),
            rgb_fps=float(args.webrtc_rgb_fps),
            depth_fps=float(args.webrtc_depth_fps),
            telemetry_hz=float(args.webrtc_telemetry_hz),
            poll_interval_ms=int(args.webrtc_poll_interval_ms),
        ),
    )
    web.run_app(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
