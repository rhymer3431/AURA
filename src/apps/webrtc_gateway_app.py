from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone WebRTC gateway for viewer IPC streams.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
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
    parser.add_argument("--cors-origin", action="append", default=[])
    parser.add_argument("--ice-server", action="append", default=[])
    return parser.parse_args(argv)


def create_app(
    args: argparse.Namespace | None = None,
    *,
    subscriber=None,
    session_manager=None,
):
    parsed_args = parse_args([]) if args is None else args
    from webrtc.config import WebRTCGatewayConfig, build_ice_server_configs, normalize_cors_origins
    from webrtc.gateway import WebRTCGateway

    cors_origins = normalize_cors_origins(parsed_args.cors_origin or ["*"])
    config = WebRTCGatewayConfig(
        host=str(parsed_args.host),
        port=int(parsed_args.port),
        control_endpoint=str(parsed_args.control_endpoint),
        telemetry_endpoint=str(parsed_args.telemetry_endpoint),
        shm_name=str(parsed_args.shm_name),
        shm_slot_size=int(parsed_args.shm_slot_size),
        shm_capacity=int(parsed_args.shm_capacity),
        enable_depth_track=bool(parsed_args.enable_depth_track),
        rgb_fps=float(parsed_args.rgb_fps),
        depth_fps=float(parsed_args.depth_fps),
        telemetry_hz=float(parsed_args.telemetry_hz),
        cors_origins=cors_origins,
        ice_servers=build_ice_server_configs(parsed_args.ice_server),
    )
    return WebRTCGateway(config, subscriber=subscriber, session_manager=session_manager).create_app()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from aiohttp import web
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("aiohttp is required to run the WebRTC gateway app.") from exc

    app = create_app(args)
    web.run_app(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
