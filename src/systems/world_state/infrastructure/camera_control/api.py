"""HTTP API for runtime G1 camera pitch control."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import urlparse


class CameraPitchApiServer:
    """Expose camera pitch state over a small local HTTP server."""

    def __init__(self, host: str, port: int, camera_sensor):
        self.host = str(host)
        self.port = int(port)
        self._camera_sensor = camera_sensor
        self._server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="camera-pitch-api", daemon=True)

    def _build_handler(self):
        camera_sensor = self._camera_sensor

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status_code: int, payload: dict[str, object]):
                body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
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
                return json.loads(raw.decode("utf-8"))

            def _status_payload(self) -> dict[str, object]:
                payload = dict(camera_sensor.pitch_status())
                payload["ok"] = True
                return payload

            def do_OPTIONS(self):
                self._send_json(HTTPStatus.NO_CONTENT, {})

            def do_GET(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path == "/healthz":
                    self._send_json(HTTPStatus.OK, {"ok": True, "service": "camera_pitch_api"})
                    return
                if path == "/camera/pitch":
                    self._send_json(HTTPStatus.OK, self._status_payload())
                    return
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {
                        "ok": False,
                        "error": "not_found",
                        "available_paths": ["/healthz", "/camera/pitch"],
                    },
                )

            def do_POST(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path != "/camera/pitch":
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                    return

                try:
                    payload = self._read_json_body()
                except json.JSONDecodeError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid_json: {exc}"})
                    return

                if "pitch_deg" in payload:
                    try:
                        pitch_deg = float(payload["pitch_deg"])
                    except (TypeError, ValueError):
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "pitch_deg must be numeric"})
                        return
                    target_pitch_deg = camera_sensor.set_pitch_deg(pitch_deg)
                    response = self._status_payload()
                    response["updated"] = "absolute"
                    response["target_pitch_deg"] = target_pitch_deg
                    self._send_json(HTTPStatus.OK, response)
                    return

                if "delta_deg" in payload:
                    try:
                        delta_deg = float(payload["delta_deg"])
                    except (TypeError, ValueError):
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "delta_deg must be numeric"})
                        return
                    target_pitch_deg = camera_sensor.add_pitch_deg(delta_deg)
                    response = self._status_payload()
                    response["updated"] = "relative"
                    response["target_pitch_deg"] = target_pitch_deg
                    self._send_json(HTTPStatus.OK, response)
                    return

                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": "expected JSON body with pitch_deg or delta_deg",
                    },
                )

            def log_message(self, format: str, *args):
                del format, args

        return Handler

    def start(self):
        self._thread.start()
        print(f"[INFO] Camera pitch API listening on http://{self.host}:{self.port}/camera/pitch")

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
