"""Unified runtime-status and camera pitch HTTP surface for control."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import urlparse


class RuntimeControlApiServer:
    """Expose runtime status and camera pitch over one local HTTP server."""

    def __init__(self, host: str, port: int, runtime_handler, camera_sensor=None):
        self.host = str(host)
        self.port = int(port)
        self._runtime_handler = runtime_handler
        self._camera_sensor = camera_sensor
        self._server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        self._server.daemon_threads = True
        self.port = int(self._server.server_port)
        self._thread = threading.Thread(target=self._server.serve_forever, name="runtime-control-api", daemon=True)

    def _build_handler(self):
        runtime_handler = self._runtime_handler
        camera_sensor = self._camera_sensor

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status_code: int, payload: dict[str, object]):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(int(status_code))
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

            def _runtime_status_payload(self) -> dict[str, object]:
                status_getter = getattr(runtime_handler, "runtime_status", None)
                if status_getter is None:
                    status_getter = getattr(runtime_handler, "command_api_status")
                payload = dict(status_getter())
                payload["ok"] = True
                return payload

            def _camera_status_payload(self) -> dict[str, object]:
                if camera_sensor is None:
                    return {"ok": False, "error": "camera_pitch_unavailable"}
                payload = dict(camera_sensor.pitch_status())
                payload["ok"] = True
                return payload

            def _handle_camera_post(self, payload: dict[str, object]):
                if camera_sensor is None:
                    self._send_json(HTTPStatus.NOT_IMPLEMENTED, {"ok": False, "error": "camera_pitch_unavailable"})
                    return
                if "pitch_deg" in payload:
                    try:
                        target_pitch_deg = camera_sensor.set_pitch_deg(float(payload["pitch_deg"]))
                    except (TypeError, ValueError):
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "pitch_deg must be numeric"})
                        return
                    response = self._camera_status_payload()
                    response["updated"] = "absolute"
                    response["target_pitch_deg"] = target_pitch_deg
                    self._send_json(HTTPStatus.OK, response)
                    return
                if "delta_deg" in payload:
                    try:
                        target_pitch_deg = camera_sensor.add_pitch_deg(float(payload["delta_deg"]))
                    except (TypeError, ValueError):
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "delta_deg must be numeric"})
                        return
                    response = self._camera_status_payload()
                    response["updated"] = "relative"
                    response["target_pitch_deg"] = target_pitch_deg
                    self._send_json(HTTPStatus.OK, response)
                    return
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "expected JSON body with pitch_deg or delta_deg"},
                )

            def do_OPTIONS(self):
                self._send_json(HTTPStatus.NO_CONTENT, {})

            def do_GET(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path == "/healthz":
                    self._send_json(HTTPStatus.OK, {"ok": True, "service": "runtime_control_api"})
                    return
                if path == "/runtime/status":
                    self._send_json(HTTPStatus.OK, self._runtime_status_payload())
                    return
                if path == "/camera/pitch":
                    status = self._camera_status_payload()
                    http_status = HTTPStatus.OK if bool(status.get("ok")) else HTTPStatus.NOT_IMPLEMENTED
                    self._send_json(http_status, status)
                    return
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {
                        "ok": False,
                        "error": "not_found",
                        "available_paths": [
                            "/healthz",
                            "/runtime/status",
                            "/camera/pitch",
                        ],
                    },
                )

            def do_POST(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                try:
                    payload = self._read_json_body()
                except json.JSONDecodeError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid_json: {exc}"})
                    return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                    return

                if path == "/camera/pitch":
                    self._handle_camera_post(payload)
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def log_message(self, format: str, *args):
                del format, args

        return Handler

    def start(self):
        self._thread.start()
        print(f"[INFO] Runtime control API listening on http://{self.host}:{self.port}")

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
