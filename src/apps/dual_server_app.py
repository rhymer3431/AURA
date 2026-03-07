from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-system orchestrator server (VLM -> NavDP).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--navdp-url", type=str, default="http://127.0.0.1:8888")
    parser.add_argument("--vlm-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--vlm-model", type=str, default="internvla-system2")
    parser.add_argument("--s2-mode", type=str, choices=("auto", "mock"), default="auto")
    parser.add_argument("--s1-period-sec", type=float, default=0.2)
    parser.add_argument("--s2-period-sec", type=float, default=1.0)
    parser.add_argument("--goal-ttl-sec", type=float, default=3.0)
    parser.add_argument("--traj-ttl-sec", type=float, default=1.5)
    parser.add_argument("--traj-max-stale-sec", type=float, default=4.0)
    parser.add_argument("--navdp-timeout-sec", type=float, default=5.0)
    parser.add_argument("--navdp-reset-timeout-sec", type=float, default=15.0)
    parser.add_argument("--vlm-timeout-sec", type=float, default=35.0)
    parser.add_argument("--s2-failure-backoff-max-sec", type=float, default=30.0)
    parser.add_argument("--stop-threshold", type=float, default=-3.0)
    parser.add_argument("--use-trajectory-z", action="store_true")
    parser.add_argument("--debug-log", action="store_true")
    return parser.parse_known_args(argv)[0]


def _read_rgb_bgr() -> np.ndarray:
    import cv2

    image_file = request.files.get("image")
    if image_file is None:
        raise ValueError("missing multipart file: image")
    image = Image.open(image_file.stream).convert("RGB")
    rgb = np.asarray(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _read_depth_m() -> np.ndarray:
    depth_file = request.files.get("depth")
    if depth_file is None:
        raise ValueError("missing multipart file: depth")
    depth = Image.open(depth_file.stream).convert("I")
    depth_np = np.asarray(depth, dtype=np.float32)
    return depth_np / 10000.0


def create_app(args: argparse.Namespace | None = None) -> Flask:
    from services.dual_orchestrator import DualOrchestrator, parse_json_field

    parsed_args = parse_args([]) if args is None else args
    orchestrator = DualOrchestrator(parsed_args)
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok", "service": "vlm_dual_server"})

    @app.route("/dual_reset", methods=["POST"])
    def dual_reset() -> Any:
        payload = request.get_json(silent=True) or {}
        ok, body = orchestrator.reset(payload)
        if ok:
            return jsonify(body)
        return jsonify(body), 400

    @app.route("/dual_step", methods=["POST"])
    def dual_step() -> Any:
        try:
            image_bgr = _read_rgb_bgr()
            depth_m = _read_depth_m()
            step_id = int(request.form.get("step_id", "-1"))
            cam_pos = np.asarray(parse_json_field(request.form.get("cam_pos"), [0.0, 0.0, 0.0]), dtype=np.float32)
            cam_quat = np.asarray(
                parse_json_field(request.form.get("cam_quat_wxyz"), [1.0, 0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
            sensor_meta = parse_json_field(request.form.get("sensor_meta"), {})
            events = parse_json_field(request.form.get("events"), {})
            if not isinstance(sensor_meta, dict):
                sensor_meta = {}
            if not isinstance(events, dict):
                events = {}
            body = orchestrator.step(
                image_bgr=image_bgr,
                depth_m=depth_m,
                step_id=step_id,
                cam_pos=cam_pos,
                cam_quat_wxyz=cam_quat,
                sensor_meta=sensor_meta,
                events=events,
            )
            return jsonify(body)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.route("/dual_debug_state", methods=["GET"])
    def dual_debug_state() -> Any:
        return jsonify(orchestrator.debug_state())

    return app


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(args)
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
