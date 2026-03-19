from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import platform
import subprocess
import time
from pathlib import Path
import socket
from typing import BinaryIO, Callable

from .config import DashboardBackendConfig
from .models import DashboardSessionRequest, PROCESS_NAMES, resolve_repo_path


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    script_path: Path
    args: tuple[str, ...]
    health_url: str
    debug_url: str = ""
    tcp_ready_host: str | None = None
    tcp_ready_port: int | None = None
    env: tuple[tuple[str, str], ...] = ()


@dataclass
class ManagedProcess:
    spec: ProcessSpec
    process: subprocess.Popen[bytes] | object
    started_at: float
    stdout_log: Path
    stderr_log: Path
    stdout_handle: BinaryIO | None = None
    stderr_handle: BinaryIO | None = None

    def pid(self) -> int | None:
        return int(getattr(self.process, "pid", 0) or 0) or None

    def poll(self) -> int | None:
        poll_fn = getattr(self.process, "poll", None)
        if callable(poll_fn):
            return poll_fn()
        return getattr(self.process, "returncode", None)


class ProcessManager:
    def __init__(
        self,
        config: DashboardBackendConfig,
        *,
        runner: Callable[[ProcessSpec, Path, Path], ManagedProcess] | None = None,
        startup_timeout_sec: float = 90.0,
    ) -> None:
        self.config = config
        self._runner = runner or self._default_runner
        self._startup_timeout_sec = max(float(startup_timeout_sec), 1.0)
        self._processes: dict[str, ManagedProcess] = {}
        self._planned_specs: dict[str, ProcessSpec] = {}
        self._current_request: DashboardSessionRequest | None = None
        self._session_started_at: float | None = None

    @property
    def current_request(self) -> DashboardSessionRequest | None:
        return self._current_request

    @property
    def session_started_at(self) -> float | None:
        return self._session_started_at

    async def start_session(self, request: DashboardSessionRequest) -> None:
        await self.stop_all()
        self.config.process_log_dir.mkdir(parents=True, exist_ok=True)
        specs = self._build_specs(request)
        self._planned_specs = {spec.name: spec for spec in specs}
        spec: ProcessSpec | None = None
        managed: ManagedProcess | None = None
        try:
            for spec in specs:
                managed = self._runner(spec, self.config.process_log_dir / f"{spec.name}.stdout.log", self.config.process_log_dir / f"{spec.name}.stderr.log")
                self._processes[spec.name] = managed
                await self._wait_ready(spec, managed)
        except Exception as exc:
            message = self._format_startup_error(spec, managed, exc)
            await self.stop_all()
            raise RuntimeError(message) from exc
        self._current_request = request
        self._session_started_at = time.time()

    async def stop_all(self) -> None:
        for name in reversed(PROCESS_NAMES):
            managed = self._processes.pop(name, None)
            if managed is None:
                continue
            await self._stop_process(managed)
        self._planned_specs = {}
        self._current_request = None
        self._session_started_at = None

    def snapshot(self) -> list[dict[str, object]]:
        required = set() if self._current_request is None else self._current_request.required_process_names()
        records: list[dict[str, object]] = []
        for name in PROCESS_NAMES:
            managed = self._processes.get(name)
            if managed is None:
                state = "not_required" if name not in required else "stopped"
                records.append(
                    {
                        "name": name,
                        "state": state,
                        "required": name in required,
                        "pid": None,
                        "exitCode": None,
                        "startedAt": None,
                        "healthUrl": self.service_urls(name)[0],
                        "stdoutLog": str(self.config.process_log_dir / f"{name}.stdout.log"),
                        "stderrLog": str(self.config.process_log_dir / f"{name}.stderr.log"),
                    }
                )
                continue
            return_code = managed.poll()
            records.append(
                {
                    "name": name,
                    "state": "running" if return_code is None else "exited",
                    "required": name in required,
                    "pid": managed.pid(),
                    "exitCode": return_code,
                    "startedAt": managed.started_at,
                    "healthUrl": managed.spec.health_url,
                    "stdoutLog": str(managed.stdout_log),
                    "stderrLog": str(managed.stderr_log),
                }
            )
        return records

    def _build_specs(self, request: DashboardSessionRequest) -> list[ProcessSpec]:
        scripts_dir = resolve_repo_path(self.config.repo_root, "scripts")
        navdp_port = self._reserve_port("127.0.0.1", 8888)
        system2_port = self._reserve_port("127.0.0.1", 8080, reserved={navdp_port})
        dual_port = self._reserve_port("127.0.0.1", 8890, reserved={navdp_port, system2_port})
        navdp_base_url = f"http://127.0.0.1:{navdp_port}"
        system2_base_url = f"http://127.0.0.1:{system2_port}"
        dual_base_url = f"http://127.0.0.1:{dual_port}"
        runtime_args = self._runtime_args(request, navdp_base_url=navdp_base_url, dual_base_url=dual_base_url)
        specs = [
            ProcessSpec(
                name="navdp",
                script_path=scripts_dir / "run_system.ps1",
                args=("-Component", "nav"),
                health_url=f"{navdp_base_url}/health",
                debug_url=f"{navdp_base_url}/debug_last_input",
                tcp_ready_host="127.0.0.1",
                tcp_ready_port=navdp_port,
                env=(("NAVDP_PORT", str(navdp_port)),),
            )
        ]
        specs.extend(
            [
                ProcessSpec(
                    name="system2",
                    script_path=scripts_dir / "run_system.ps1",
                    args=("-Component", "s2"),
                    health_url=system2_base_url,
                    tcp_ready_host="127.0.0.1",
                    tcp_ready_port=system2_port,
                    env=(("INTERNVLA_HOST", "127.0.0.1"), ("INTERNVLA_PORT", str(system2_port))),
                ),
                ProcessSpec(
                    name="dual",
                    script_path=scripts_dir / "run_system.ps1",
                    args=("-Component", "dual"),
                    health_url=f"{dual_base_url}/health",
                    debug_url=f"{dual_base_url}/dual_debug_state",
                    tcp_ready_host="127.0.0.1",
                    tcp_ready_port=dual_port,
                    env=(
                        ("DUAL_SERVER_HOST", "127.0.0.1"),
                        ("DUAL_SERVER_PORT", str(dual_port)),
                        ("DUAL_NAVDP_URL", navdp_base_url),
                        ("DUAL_VLM_URL", system2_base_url),
                    ),
                ),
            ]
        )
        specs.append(
            ProcessSpec(
                name="runtime",
                script_path=scripts_dir / "run_system.ps1",
                args=("-Component", "runtime", *runtime_args),
                health_url="",
                env=(
                    ("G1_POINTGOAL_SCENE_PRESET", request.scene_preset),
                    ("AURA_RUNTIME_TRACE_PATH", str(self.config.process_log_dir / "runtime.trace.log")),
                ),
            )
        )
        return specs

    def _runtime_args(
        self,
        request: DashboardSessionRequest,
        *,
        navdp_base_url: str,
        dual_base_url: str,
    ) -> list[str]:
        args = [
            "--native-viewer",
            "off",
            "--server-url",
            navdp_base_url,
            "--dual-server-url",
            dual_base_url,
        ]
        if request.launch_mode == "gui":
            args += ["--launch-mode", "gui"]
        else:
            args += ["--headless"]
        if request.viewer_enabled:
            args += ["--viewer-publish"]
        else:
            args += ["--no-viewer-publish"]
        if not request.memory_store:
            args += ["--no-memory-store"]
        if not request.detection_enabled:
            args += ["--skip-detection"]
        args += [
            "--action-scale",
            str(request.locomotion_config.action_scale),
            "--onnx-device",
            request.locomotion_config.onnx_device,
            "--cmd-max-vx",
            str(request.locomotion_config.cmd_max_vx),
            "--cmd-max-vy",
            str(request.locomotion_config.cmd_max_vy),
            "--cmd-max-wz",
            str(request.locomotion_config.cmd_max_wz),
        ]
        return args

    def service_urls(self, name: str) -> tuple[str, str]:
        managed = self._processes.get(name)
        if managed is not None:
            return managed.spec.health_url, managed.spec.debug_url
        planned = self._planned_specs.get(name)
        if planned is not None:
            return planned.health_url, planned.debug_url
        if name == "navdp":
            return "http://127.0.0.1:8888/health", "http://127.0.0.1:8888/debug_last_input"
        if name == "dual":
            return "http://127.0.0.1:8890/health", "http://127.0.0.1:8890/dual_debug_state"
        if name == "system2":
            return "http://127.0.0.1:8080", ""
        return "", ""

    async def _wait_ready(self, spec: ProcessSpec, managed: ManagedProcess) -> None:
        if spec.tcp_ready_host is None or spec.tcp_ready_port is None:
            await asyncio.sleep(0.5)
            return
        deadline = time.time() + self._startup_timeout_sec
        while time.time() < deadline:
            if managed.poll() is not None:
                raise RuntimeError(f"{spec.name} exited before becoming ready")
            if await asyncio.to_thread(self._tcp_ready, spec.tcp_ready_host, spec.tcp_ready_port):
                return
            await asyncio.sleep(0.5)
        raise RuntimeError(f"{spec.name} did not become ready in time")

    async def _stop_process(self, managed: ManagedProcess) -> None:
        process = managed.process
        if managed.poll() is None:
            stopped = False
            pid = managed.pid()
            if pid is not None and self._should_kill_process_tree():
                stopped = await asyncio.to_thread(self._kill_process_tree, pid)
            if not stopped:
                terminate = getattr(process, "terminate", None)
                if callable(terminate):
                    terminate()
                wait_fn = getattr(process, "wait", None)
                if callable(wait_fn):
                    try:
                        await asyncio.to_thread(wait_fn, 5.0)
                        stopped = True
                    except Exception:
                        kill = getattr(process, "kill", None)
                        if callable(kill):
                            kill()
            if stopped:
                wait_fn = getattr(process, "wait", None)
                if callable(wait_fn):
                    try:
                        await asyncio.to_thread(wait_fn, 5.0)
                    except Exception:
                        pass
        if managed.stdout_handle is not None:
            managed.stdout_handle.close()
        if managed.stderr_handle is not None:
            managed.stderr_handle.close()

    @staticmethod
    def _should_kill_process_tree() -> bool:
        return platform.system().lower() == "windows"

    @staticmethod
    def _kill_process_tree(pid: int) -> bool:
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return False
        return completed.returncode == 0

    @staticmethod
    def _tcp_ready(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            return False

    def _default_health_url(self, name: str) -> str:
        return self.service_urls(name)[0]

    @staticmethod
    def _reserve_port(host: str, preferred_port: int, *, reserved: set[int] | None = None) -> int:
        reserved_ports = set() if reserved is None else set(reserved)
        if preferred_port not in reserved_ports and ProcessManager._can_bind(host, preferred_port):
            return preferred_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _can_bind(host: str, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
                return True
        except OSError:
            return False

    @staticmethod
    def _tail_file(path: Path, *, limit: int = 4) -> str:
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return " | ".join(line.strip() for line in lines[-max(limit, 1) :] if line.strip())

    def _format_startup_error(self, spec: ProcessSpec | None, managed: ManagedProcess | None, exc: Exception) -> str:
        if spec is None:
            return f"process startup failed: {exc}"
        if managed is None:
            return f"{spec.name} failed to start: {exc}"
        stderr_tail = self._tail_file(managed.stderr_log)
        if stderr_tail == "":
            return f"{spec.name} failed to start: {exc}"
        return f"{spec.name} failed to start: {exc}. stderr: {stderr_tail}"

    def _default_runner(self, spec: ProcessSpec, stdout_log: Path, stderr_log: Path) -> ManagedProcess:
        stdout_handle = stdout_log.open("wb")
        stderr_handle = stderr_log.open("wb")
        child_env = os.environ.copy()
        for key, value in spec.env:
            child_env[str(key)] = str(value)
        src_dir = str(resolve_repo_path(self.config.repo_root, "src"))
        existing_pythonpath = [entry for entry in str(child_env.get("PYTHONPATH", "")).split(os.pathsep) if entry.strip()]
        child_env["PYTHONPATH"] = os.pathsep.join([src_dir, *[entry for entry in existing_pythonpath if entry != src_dir]])
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(spec.script_path),
                *spec.args,
            ],
            cwd=str(self.config.repo_root),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=child_env,
        )
        return ManagedProcess(
            spec=spec,
            process=process,
            started_at=time.time(),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )
