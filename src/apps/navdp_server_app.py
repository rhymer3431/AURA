from __future__ import annotations

import argparse
from typing import Any

from flask import Flask, jsonify, request


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/PJLAB/caiwenzhe/Desktop/navdp_bench/baselines/navdp/checkpoints/cross-waic-final4-125.ckpt",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", type=str, choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--tf32", dest="tf32", action="store_true")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.set_defaults(amp=True, tf32=True)
    return parser.parse_known_args(argv)[0]


def create_app(args: argparse.Namespace | None = None) -> Flask:
    from services.navdp_inference_service import NavDPInferenceService

    parsed_args = parse_args([]) if args is None else args
    service = NavDPInferenceService(
        checkpoint=parsed_args.checkpoint,
        device=parsed_args.device,
        amp=parsed_args.amp,
        amp_dtype=parsed_args.amp_dtype,
        tf32=parsed_args.tf32,
    )
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    @app.route("/healthz", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok", "service": "navdp_server"})

    @app.route("/navigator_reset", methods=["POST"])
    def navigator_reset() -> Any:
        return jsonify(service.navigator_reset(request.get_json()))

    @app.route("/navigator_reset_env", methods=["POST"])
    def navigator_reset_env() -> Any:
        return jsonify(service.navigator_reset_env(request.get_json()))

    @app.route("/pointgoal_step", methods=["POST"])
    def pointgoal_step() -> Any:
        goal_data = service.parse_goal_data(request.form.get("goal_data"))
        body = service.pointgoal_step(request.files["image"], request.files["depth"], goal_data)
        return jsonify(body)

    @app.route("/debug_last_input", methods=["GET"])
    def debug_last_input() -> Any:
        return jsonify({"last_input_meta": service.last_input_meta})

    @app.route("/pixelgoal_step", methods=["POST"])
    def pixelgoal_step() -> Any:
        goal_data = service.parse_goal_data(request.form.get("goal_data"))
        body = service.pixelgoal_step(request.files["image"], request.files["depth"], goal_data)
        return jsonify(body)

    @app.route("/imagegoal_step", methods=["POST"])
    def imagegoal_step() -> Any:
        body = service.imagegoal_step(request.files["image"], request.files["depth"], request.files["goal"])
        return jsonify(body)

    @app.route("/nogoal_step", methods=["POST"])
    def nogoal_step() -> Any:
        body = service.nogoal_step(request.files["image"], request.files["depth"])
        return jsonify(body)

    @app.route("/navdp_step_ip_mixgoal", methods=["POST"])
    def point_image_mixgoal_step() -> Any:
        goal_data = service.parse_goal_data(request.form.get("goal_data"))
        body = service.point_image_mixgoal_step(
            request.files["image"],
            request.files["depth"],
            request.files["image_goal"],
            goal_data,
        )
        return jsonify(body)

    return app


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(args)
    app.run(host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
