"""Standalone planner-system service."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any
from urllib.parse import urlparse

from systems.inference.api.planner import make_http_completion
from systems.navigation.api.runtime import NavigationSystemClient
from systems.planner.tasking.aura_adapter import AuraTaskingAdapter
from systems.planner.tasking.reporting import build_navigation_instruction


@dataclass(slots=True)
class PlannerTaskState:
    task_id: str
    instruction: str
    language: str
    task_frame: dict[str, Any]
    subgoals: list[dict[str, Any]]
    current_subgoal_index: int
    status: str
    started_at: float
    last_error: str | None = None

    @property
    def current_subgoal(self) -> dict[str, Any] | None:
        if self.current_subgoal_index < 0 or self.current_subgoal_index >= len(self.subgoals):
            return None
        return self.subgoals[self.current_subgoal_index]


class PlannerSystem:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        completion = None
        planner_model_base_url = str(args.planner_model_base_url).strip()
        if planner_model_base_url:
            completion = make_http_completion(planner_model_base_url)
        self._adapter = AuraTaskingAdapter(
            completion=completion,
            model=str(args.planner_model),
            timeout=float(args.planner_timeout),
        )
        self._navigation = NavigationSystemClient(str(args.navigation_url), timeout_s=float(args.navigation_timeout))
        self._lock = threading.Lock()
        self._task: PlannerTaskState | None = None

    def submit_task(self, instruction: str, language: str, *, task_id: str | None = None) -> dict[str, object]:
        normalized_instruction = " ".join(str(instruction).strip().split())
        if not normalized_instruction:
            raise ValueError("instruction must be a non-empty string")
        normalized_task_id = str(task_id or "").strip()

        task_frame = self._adapter.plan_task_frame(normalized_instruction)
        subgoals = self._adapter.initialize_subgoals(task_frame)
        task = PlannerTaskState(
            task_id=normalized_task_id or f"planner-{time.time_ns()}",
            instruction=normalized_instruction,
            language=str(language).strip() or "auto",
            task_frame=task_frame,
            subgoals=subgoals,
            current_subgoal_index=0 if subgoals else -1,
            status="running" if subgoals else "idle",
            started_at=time.time(),
        )

        current_subgoal = task.current_subgoal
        if current_subgoal is not None and current_subgoal.get("type") == "navigate":
            nav_instruction = build_navigation_instruction(current_subgoal["input"]["target"], language="en")
            self._navigation.command(nav_instruction, "en", task_id=task.task_id)
        else:
            try:
                self._navigation.cancel()
            except Exception:
                pass

        with self._lock:
            self._task = task
        return self.status_payload()

    def cancel(self) -> dict[str, object]:
        with self._lock:
            task = self._task
            if task is not None:
                task.status = "cancelled"
        try:
            self._navigation.cancel()
        except Exception:
            pass
        return self.status_payload()

    def status_payload(self) -> dict[str, object]:
        with self._lock:
            task = self._task
        if task is None:
            return {
                "ok": True,
                "service": "planner_system",
                "task_status": "idle",
                "task_id": None,
                "instruction": "",
                "language": "auto",
                "task_frame": None,
                "current_subgoal": None,
                "subgoals": [],
                "started_at": None,
                "last_error": None,
            }
        return {
            "ok": True,
            "service": "planner_system",
            "task_status": task.status,
            "task_id": task.task_id,
            "instruction": task.instruction,
            "language": task.language,
            "task_frame": task.task_frame,
            "current_subgoal": task.current_subgoal,
            "subgoals": task.subgoals,
            "started_at": task.started_at,
            "last_error": task.last_error,
        }


class PlannerSystemServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._service = PlannerSystem(args)
        self._server = ThreadingHTTPServer((str(args.host), int(args.port)), self._build_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="planner-system-api", daemon=True)

    def _build_handler(self):
        service = self._service

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
                raw = self.rfile.read(content_length) if content_length > 0 else b""
                if not raw:
                    return {}
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("expected JSON object body")
                return payload

            def do_OPTIONS(self):
                self._send_json(HTTPStatus.NO_CONTENT, {})

            def do_GET(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path == "/healthz":
                    self._send_json(HTTPStatus.OK, {"ok": True, "service": "planner_system"})
                    return
                if path == "/planner/status":
                    self._send_json(HTTPStatus.OK, service.status_payload())
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

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

                try:
                    if path == "/planner/task":
                        instruction = payload.get("instruction")
                        language = payload.get("language", "auto")
                        task_id = payload.get("task_id")
                        if not isinstance(instruction, str) or not instruction.strip():
                            raise ValueError("instruction must be a non-empty string")
                        if not isinstance(language, str):
                            raise ValueError("language must be a string")
                        if task_id is not None and not isinstance(task_id, str):
                            raise ValueError("task_id must be a string")
                        self._send_json(HTTPStatus.OK, service.submit_task(instruction, language, task_id=task_id))
                        return
                    if path == "/planner/cancel":
                        self._send_json(HTTPStatus.OK, service.cancel())
                        return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def log_message(self, format: str, *args):
                del format, args

        return Handler

    def start(self) -> None:
        self._thread.start()
        print(f"[INFO] Planner system API listening on http://{self.args.host}:{self.args.port}")

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the planner system service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17881)
    parser.add_argument("--navigation-url", default="http://127.0.0.1:17882")
    parser.add_argument("--navigation-timeout", type=float, default=5.0)
    parser.add_argument("--planner-model-base-url", default="http://127.0.0.1:8093/v1/chat/completions")
    parser.add_argument("--planner-model", default="Qwen3-1.7B-Q4_K_M-Instruct.gguf")
    parser.add_argument("--planner-timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    server = PlannerSystemServer(args)
    server.start()
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
