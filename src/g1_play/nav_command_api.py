"""HTTP API for runtime InternVLA navigation command updates."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import urlparse


class RuntimeNavCommandApiServer:
    """Expose runtime natural-language command updates over a local HTTP server."""

    def __init__(self, host: str, port: int, command_handler):
        self.host = str(host)
        self.port = int(port)
        self._command_handler = command_handler
        self._server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        self._server.daemon_threads = True
        self.port = int(self._server.server_port)
        self._thread = threading.Thread(target=self._server.serve_forever, name="runtime-nav-command-api", daemon=True)

    def _build_handler(self):
        command_handler = self._command_handler

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status_code: int, payload: dict[str, object]):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def _read_json_body(self) -> dict[str, object]:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    return {}
                raw = self.rfile.read(content_length)
                if not raw:
                    return {}
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object body")
                return payload

            def _status_payload(self) -> dict[str, object]:
                payload = dict(command_handler.command_api_status())
                payload["ok"] = True
                return payload

            def do_OPTIONS(self):
                self._send_json(HTTPStatus.NO_CONTENT, {})

            def do_GET(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path == "/healthz":
                    self._send_json(HTTPStatus.OK, {"ok": True, "service": "runtime_nav_command_api"})
                    return
                if path == "/nav/command":
                    self._send_json(HTTPStatus.OK, self._status_payload())
                    return
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {
                        "ok": False,
                        "error": "not_found",
                        "available_paths": ["/healthz", "/nav/command"],
                    },
                )

            def do_POST(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path != "/nav/command":
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                    return
                try:
                    payload = self._read_json_body()
                except json.JSONDecodeError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid_json: {exc}"})
                    return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                    return

                instruction = payload.get("instruction")
                language = payload.get("language")
                if not isinstance(instruction, str) or not instruction.strip():
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "instruction must be a non-empty string"})
                    return
                if language is not None and not isinstance(language, str):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "language must be a string when provided"})
                    return

                try:
                    response = dict(
                        command_handler.apply_runtime_command(
                            instruction=instruction,
                            language=language,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    )
                    return

                response["ok"] = True
                self._send_json(HTTPStatus.OK, response)

            def log_message(self, format: str, *args):
                del format, args

        return Handler

    def start(self):
        self._thread.start()
        print(f"[INFO] Runtime nav command API listening on http://{self.host}:{self.port}/nav/command")

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
