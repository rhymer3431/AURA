from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from .schema import generate_stub_plan, validate_plan_payload


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [planner_server] %(message)s",
    )


class PlannerHttpServer(ThreadingHTTPServer):
    def __init__(self, *args: Any, use_mock: bool = True, model_path: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_mock = use_mock
        self.model_path = model_path


class PlannerHandler(BaseHTTPRequestHandler):
    server: PlannerHttpServer  # type: ignore[assignment]

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.client_address[0], fmt % args)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/plan":
            self._write_json(404, {"error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            request_payload = json.loads(raw or "{}")
            user_command = str(request_payload.get("user_command", ""))
            world_state = request_payload.get("world_state", {})
        except Exception as exc:
            self._write_json(400, {"error": f"invalid_request:{exc}"})
            return

        try:
            if self.server.use_mock:
                plan_payload = generate_stub_plan(user_command, world_state)
            else:
                # Current non-mock path uses deterministic rule planning.
                # TODO: Plug in Nanbeige4.1-3B INT4 model inference.
                plan_payload = generate_stub_plan(user_command, world_state)
                plan_payload["notes"] = "heuristic planner output (non-mock mode)"

            validate_plan_payload(plan_payload)
            self._write_json(200, plan_payload)
        except Exception as exc:
            logging.exception("Planner error")
            self._write_json(500, {"error": f"planner_failed:{exc}"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local planner server (Nanbeige INT4 / mock).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--model-path", type=str, default="models/nanbeige4_1_3b_int4")
    parser.add_argument("--mock", action="store_true", help="Force mock planner output.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    server = PlannerHttpServer(
        (args.host, args.port),
        PlannerHandler,
        use_mock=bool(args.mock),
        model_path=args.model_path,
    )
    logging.info(
        "Planner server started at http://%s:%s (mock=%s, model=%s)",
        args.host,
        args.port,
        bool(args.mock),
        args.model_path,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logging.info("Planner server stopped.")


if __name__ == "__main__":
    main()
