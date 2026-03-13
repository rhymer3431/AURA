from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local dashboard backend for AURA runtime control and streaming.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--dashboard-dir", type=str, default="dashboard")
    parser.add_argument("--dev-origin", type=str, default="http://127.0.0.1:5173")
    parser.add_argument("--control-endpoint", type=str, default="tcp://127.0.0.1:5580")
    parser.add_argument("--telemetry-endpoint", type=str, default="tcp://127.0.0.1:5581")
    parser.add_argument("--shm-name", type=str, default="g1_view_frames")
    parser.add_argument("--shm-slot-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--shm-capacity", type=int, default=8)
    parser.add_argument("--enable-depth-track", dest="enable_depth_track", action="store_true")
    parser.add_argument("--disable-depth-track", dest="enable_depth_track", action="store_false")
    parser.set_defaults(enable_depth_track=True)
    parser.add_argument("--rgb-fps", type=float, default=15.0)
    parser.add_argument("--depth-fps", type=float, default=5.0)
    parser.add_argument("--telemetry-hz", type=float, default=10.0)
    parser.add_argument("--ice-server", action="append", default=[])
    return parser.parse_args(argv)


def create_app(args: argparse.Namespace | None = None):
    parsed_args = parse_args([]) if args is None else args
    from dashboard_backend.app import DashboardWebApp
    from dashboard_backend.config import DashboardBackendConfig
    from webrtc.config import build_ice_server_configs

    config = DashboardBackendConfig(
        host=str(parsed_args.host),
        port=int(parsed_args.port),
        repo_root=Path(__file__).resolve().parents[2],
        dashboard_dir=(Path(__file__).resolve().parents[2] / str(parsed_args.dashboard_dir)).resolve(),
        dev_origin=str(parsed_args.dev_origin),
        control_endpoint=str(parsed_args.control_endpoint),
        telemetry_endpoint=str(parsed_args.telemetry_endpoint),
        shm_name=str(parsed_args.shm_name),
        shm_slot_size=int(parsed_args.shm_slot_size),
        shm_capacity=int(parsed_args.shm_capacity),
        enable_depth_track=bool(parsed_args.enable_depth_track),
        rgb_fps=float(parsed_args.rgb_fps),
        depth_fps=float(parsed_args.depth_fps),
        telemetry_hz=float(parsed_args.telemetry_hz),
        ice_servers=build_ice_server_configs(parsed_args.ice_server),
    )
    return DashboardWebApp(config).create_app()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from aiohttp import web

    app = create_app(args)
    web.run_app(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
