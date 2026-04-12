"""Launch and supervise the runtime stack through the canonical launcher scripts."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from systems.shared.contracts.dashboard import LogRecord, ProcessRecord


CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _now_ns() -> int:
    return int(time.time() * 1_000_000_000)


def _event(message: str, *, level: str = "info") -> dict[str, object]:
    return LogRecord(
        source="runtime",
        stream="event",
        level=level,
        message=message,
        timestampNs=_now_ns(),
    ).to_dict()


def _json_get(url: str, *, timeout_s: float) -> tuple[bool, str | None]:
    request = Request(str(url).rstrip("/"), headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return response.status < 500, None
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return False, f"http_{exc.code}: {detail}"
    except URLError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _load_json_from_stdout(stdout: str) -> dict[str, object]:
    lines = [line.strip() for line in str(stdout).splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("launcher stdout was empty")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise RuntimeError("launcher did not emit a JSON object")
    return payload


@dataclass(slots=True)
class LauncherSpec:
    name: str
    script_path: Path
    env: dict[str, str]
    health_url: str
    endpoints: dict[str, object]
    required: bool = True
    wait_for_health: bool = True
    start_timeout_s: float = 30.0


@dataclass(slots=True)
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]
    stdout_log: Path
    stderr_log: Path
    started_at: float
    health_url: str
    required: bool

    def snapshot(self) -> dict[str, object]:
        return ProcessRecord(
            name=self.name,
            state="running" if self.process.poll() is None else "exited",
            required=self.required,
            pid=self.process.pid,
            exitCode=self.process.poll(),
            startedAt=self.started_at,
            healthUrl=self.health_url,
            stdoutLog=str(self.stdout_log),
            stderrLog=str(self.stderr_log),
        ).to_dict()


class RuntimeProcessRegistry:
    """Track launcher-owned processes and their log files."""

    def __init__(self, repo_root: Path):
        self._repo_root = Path(repo_root)
        self._log_dir = self._repo_root / "logs" / "runtime"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, ManagedProcess] = {}

    def start(self, spec: LauncherSpec) -> ManagedProcess:
        current = self._processes.get(spec.name)
        if current is not None and current.process.poll() is None:
            return current
        stdout_log = self._log_dir / f"{spec.name}.stdout.log"
        stderr_log = self._log_dir / f"{spec.name}.stderr.log"
        stdout_handle = open(stdout_log, "a", encoding="utf-8")
        stderr_handle = open(stderr_log, "a", encoding="utf-8")
        command = ["cmd.exe", "/d", "/c", str(spec.script_path)]
        process = subprocess.Popen(
            command,
            cwd=str(self._repo_root),
            env=spec.env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            creationflags=CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        managed = ManagedProcess(
            name=spec.name,
            process=process,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            started_at=time.time(),
            health_url=spec.health_url,
            required=spec.required,
        )
        self._processes[spec.name] = managed
        return managed

    def stop(self, name: str, *, timeout_s: float = 5.0) -> None:
        managed = self._processes.get(name)
        if managed is None or managed.process.poll() is not None:
            return
        if os.name == "nt":
            completed = subprocess.run(
                ["taskkill", "/PID", str(managed.process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=max(timeout_s, 1.0),
                check=False,
            )
            if completed.returncode not in (0, 128):
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "taskkill failed")
            return
        managed.process.terminate()
        try:
            managed.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            managed.process.kill()
            managed.process.wait(timeout=timeout_s)

    def stop_many(self, names: list[str]) -> None:
        for name in reversed(names):
            try:
                self.stop(name)
            except Exception:
                continue

    def snapshots(self) -> list[dict[str, object]]:
        return [self._processes[name].snapshot() for name in sorted(self._processes.keys())]


def _launcher_json(script_path: Path, env: Mapping[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(script_path), "-PrintConfigJson"],
        cwd=str(script_path.parents[2]),
        env=dict(env),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"{script_path.name} -PrintConfigJson failed: {detail}")
    return _load_json_from_stdout(completed.stdout)


def _coerce_number(config: dict[str, Any], key: str, fallback: float) -> str:
    value = config.get(key, fallback)
    return str(value)


def _resolve_service_url(
    env: Mapping[str, str],
    *,
    explicit_key: str,
    host_key: str,
    port_key: str,
    default_host: str,
    default_port: int,
    suffix: str = "",
) -> str:
    explicit = str(env.get(explicit_key, "")).strip()
    if explicit:
        return explicit.rstrip("/")
    host = str(env.get(host_key, default_host)).strip() or default_host
    port = int(str(env.get(port_key, default_port)).strip() or default_port)
    return f"http://{host}:{port}{suffix}".rstrip("/")


def _bool_env(value: Any, *, default: bool) -> str:
    if value is None:
        return "1" if default else "0"
    return "1" if bool(value) else "0"


def build_default_launchers(
    repo_root: Path,
    session_config: dict[str, Any],
    base_env: Mapping[str, str],
) -> list[LauncherSpec]:
    scripts_root = Path(repo_root) / "scripts" / "run_system"
    navdp_url = _resolve_service_url(
        base_env,
        explicit_key="NAVDP_URL",
        host_key="NAVDP_HOST",
        port_key="NAVDP_PORT",
        default_host="127.0.0.1",
        default_port=18888,
    )
    system2_url = _resolve_service_url(
        base_env,
        explicit_key="SYSTEM2_URL",
        host_key="SYSTEM2_HOST",
        port_key="SYSTEM2_PORT",
        default_host="127.0.0.1",
        default_port=15801,
    )
    planner_model_base_url = _resolve_service_url(
        base_env,
        explicit_key="PLANNER_MODEL_BASE_URL",
        host_key="PLANNER_MODEL_HOST",
        port_key="PLANNER_MODEL_PORT",
        default_host="127.0.0.1",
        default_port=8093,
        suffix="/v1/chat/completions",
    )

    navigation_script = scripts_root / "navigation_system_windows.bat"
    navigation_env = {
        **dict(base_env),
        "SYSTEM2_URL": system2_url,
        "NAVDP_URL": navdp_url,
        "NAVIGATION_BACKEND_AUTOSTART": str(base_env.get("NAVIGATION_BACKEND_AUTOSTART", "1")),
    }
    navigation_config = _launcher_json(navigation_script, navigation_env)
    navigation_url = str(navigation_config["navigation_system_url"]).rstrip("/")

    planner_script = scripts_root / "planner_system_windows.bat"
    planner_env = {
        **dict(base_env),
        "NAVIGATION_SYSTEM_URL": navigation_url,
        "PLANNER_MODEL_BASE_URL": planner_model_base_url,
    }
    planner_config = _launcher_json(planner_script, planner_env)
    planner_url = str(planner_config["planner_system_url"]).rstrip("/")

    control_script = scripts_root / "control_runtime_windows.bat"
    locomotion = session_config.get("locomotionConfig") if isinstance(session_config.get("locomotionConfig"), dict) else {}
    control_env = {
        **dict(base_env),
        "NAVIGATION_URL": navigation_url,
        "AURA_LAUNCH_MODE": str(session_config.get("launchMode", "gui")),
        "AURA_VIEWER_ENABLED": _bool_env(session_config.get("viewerEnabled"), default=True),
        "AURA_MEMORY_STORE": _bool_env(session_config.get("memoryStore"), default=False),
        "AURA_DETECTION_ENABLED": _bool_env(session_config.get("detectionEnabled"), default=True),
        "AURA_ACTION_SCALE": _coerce_number(locomotion, "actionScale", 0.5),
        "AURA_ONNX_DEVICE": str(locomotion.get("onnxDevice", "auto")),
        "AURA_CMD_MAX_VX": _coerce_number(locomotion, "cmdMaxVx", 0.5),
        "AURA_CMD_MAX_VY": _coerce_number(locomotion, "cmdMaxVy", 0.3),
        "AURA_CMD_MAX_WZ": _coerce_number(locomotion, "cmdMaxWz", 0.8),
    }
    control_config = _launcher_json(control_script, control_env)
    control_url = str(control_config["runtime_control_api_url"]).rstrip("/")

    return [
        LauncherSpec(
            name="navigation_system",
            script_path=navigation_script,
            env=navigation_env,
            health_url=f"{navigation_url}/healthz",
            endpoints={
                "navigationSystemUrl": navigation_url,
                "system2Url": system2_url,
                "navdpUrl": navdp_url,
            },
            wait_for_health=False,
            start_timeout_s=30.0,
        ),
        LauncherSpec(
            name="planner_system",
            script_path=planner_script,
            env=planner_env,
            health_url=f"{planner_url}/healthz",
            endpoints={"plannerSystemUrl": planner_url},
            wait_for_health=False,
            start_timeout_s=30.0,
        ),
        LauncherSpec(
            name="control_runtime",
            script_path=control_script,
            env=control_env,
            health_url=f"{control_url}/healthz",
            endpoints={"controlRuntimeUrl": control_url},
            start_timeout_s=60.0,
        ),
    ]


class RuntimeService:
    """Lifecycle owner for the runtime stack."""

    def __init__(
        self,
        repo_root: Path,
        *,
        base_env: Mapping[str, str] | None = None,
        launcher_factory: Callable[[Path, dict[str, Any], Mapping[str, str]], list[LauncherSpec]] | None = None,
    ):
        self._repo_root = Path(repo_root)
        self._base_env = dict(os.environ if base_env is None else base_env)
        self._launcher_factory = build_default_launchers if launcher_factory is None else launcher_factory
        self._registry = RuntimeProcessRegistry(self._repo_root)
        self._lock = threading.RLock()
        self._session_state = "inactive"
        self._session_config: dict[str, Any] | None = None
        self._started_at: float | None = None
        self._service_endpoints: dict[str, object] = {}
        self._last_error: str | None = None
        self._last_event: dict[str, object] | None = _event("runtime initialized")

    def _wait_for_health(self, health_url: str, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        request_timeout_s = min(max(timeout_s / 4.0, 2.0), 10.0)
        while time.monotonic() < deadline:
            remaining_s = max(deadline - time.monotonic(), 1.0)
            ok, _error = _json_get(health_url, timeout_s=min(request_timeout_s, remaining_s))
            if ok:
                return
            time.sleep(0.5)
        raise TimeoutError(f"timed out waiting for {health_url}")

    def _set_failure(self, message: str) -> dict[str, object]:
        self._session_state = "inactive"
        self._session_config = None
        self._started_at = None
        self._last_error = message
        self._last_event = _event(message, level="error")
        return self.state_payload(ok=False)

    def start_session(self, session_config: dict[str, Any]) -> dict[str, object]:
        with self._lock:
            if self._session_state == "running":
                return self.state_payload()
            self._session_state = "starting"
            self._last_error = None
            self._last_event = _event("starting runtime session")
            try:
                launchers = self._launcher_factory(self._repo_root, dict(session_config), self._base_env)
                started_names: list[str] = []
                service_endpoints: dict[str, object] = {}
                for launcher in launchers:
                    self._registry.start(launcher)
                    started_names.append(launcher.name)
                    if launcher.wait_for_health:
                        self._wait_for_health(launcher.health_url, timeout_s=launcher.start_timeout_s)
                    service_endpoints.update(launcher.endpoints)
                self._session_state = "running"
                self._session_config = dict(session_config)
                self._started_at = time.time()
                self._service_endpoints = service_endpoints
                self._last_event = _event("runtime session running")
                return self.state_payload()
            except Exception as exc:  # noqa: BLE001
                self._registry.stop_many(started_names if "started_names" in locals() else [])
                return self._set_failure(f"{type(exc).__name__}: {exc}")

    def stop_session(self) -> dict[str, object]:
        with self._lock:
            self._session_state = "stopping"
            self._registry.stop_many(["control_runtime", "planner_system", "navigation_system"])
            self._session_state = "inactive"
            self._session_config = None
            self._started_at = None
            self._last_error = None
            self._last_event = _event("runtime session stopped")
            return self.state_payload()

    def state_payload(self, *, ok: bool = True) -> dict[str, object]:
        with self._lock:
            return {
                "ok": ok,
                "session": {
                    "active": self._session_state == "running",
                    "state": self._session_state,
                    "startedAt": self._started_at,
                    "config": None if self._session_config is None else dict(self._session_config),
                    "lastEvent": self._last_event,
                },
                "processes": self._registry.snapshots(),
                "serviceEndpoints": dict(self._service_endpoints),
                "lastError": self._last_error,
            }


class RuntimeServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._service = RuntimeService(Path(args.repo_root))
        self._server = ThreadingHTTPServer((str(args.host), int(args.port)), self._build_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="runtime-api", daemon=True)

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
                    state = service.state_payload()
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "service": "runtime",
                            "sessionState": state["session"]["state"],
                        },
                    )
                    return
                if path == "/session/state":
                    self._send_json(HTTPStatus.OK, service.state_payload())
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

                if path == "/session/start":
                    response = service.start_session(payload)
                    status = HTTPStatus.OK if bool(response.get("ok")) else HTTPStatus.SERVICE_UNAVAILABLE
                    self._send_json(status, response)
                    return
                if path == "/session/stop":
                    self._send_json(HTTPStatus.OK, service.stop_session())
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def log_message(self, format: str, *args):
                del format, args

        return Handler

    def start(self) -> None:
        self._thread.start()
        print(f"[INFO] Runtime API listening on http://{self.args.host}:{self.args.port}")

    def shutdown(self) -> None:
        self._service.stop_session()
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the runtime service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18096)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    server = RuntimeServer(args)
    server.start()
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
