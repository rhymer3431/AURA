from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    tcp_ready_host: str | None = None
    tcp_ready_port: int | None = None


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
        try:
            for spec in specs:
                managed = self._runner(spec, self.config.process_log_dir / f"{spec.name}.stdout.log", self.config.process_log_dir / f"{spec.name}.stderr.log")
                self._processes[spec.name] = managed
                await self._wait_ready(spec, managed)
        except Exception:
            await self.stop_all()
            raise
        self._current_request = request
        self._session_started_at = time.time()

    async def stop_all(self) -> None:
        for name in reversed(PROCESS_NAMES):
            managed = self._processes.pop(name, None)
            if managed is None:
                continue
            await self._stop_process(managed)
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
                        "healthUrl": self._default_health_url(name),
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
        scripts_dir = resolve_repo_path(self.config.repo_root, "scripts", "powershell")
        runtime_args = self._runtime_args(request)
        specs = [
            ProcessSpec(
                name="navdp",
                script_path=scripts_dir / "run_navdp_server.ps1",
                args=(),
                health_url="http://127.0.0.1:8888/health",
                tcp_ready_host="127.0.0.1",
                tcp_ready_port=8888,
            )
        ]
        if request.planner_mode == "interactive":
            specs.extend(
                [
                    ProcessSpec(
                        name="system2",
                        script_path=scripts_dir / "run_internvla_system2.ps1",
                        args=(),
                        health_url="http://127.0.0.1:8080",
                        tcp_ready_host="127.0.0.1",
                        tcp_ready_port=8080,
                    ),
                    ProcessSpec(
                        name="dual",
                        script_path=scripts_dir / "run_dual_server.ps1",
                        args=(),
                        health_url="http://127.0.0.1:8890/health",
                        tcp_ready_host="127.0.0.1",
                        tcp_ready_port=8890,
                    ),
                ]
            )
        specs.append(
            ProcessSpec(
                name="runtime",
                script_path=scripts_dir / "run_aura_runtime.ps1",
                args=tuple(runtime_args),
                health_url="",
            )
        )
        return specs

    def _runtime_args(self, request: DashboardSessionRequest) -> list[str]:
        args = [
            "--planner-mode",
            request.planner_mode,
            "--scene-preset",
            request.scene_preset,
            "--native-viewer",
            "off",
        ]
        if request.launch_mode == "gui":
            args += ["--launch-mode", "gui"]
        else:
            args += ["--headless"]
        if request.viewer_enabled:
            args += ["--viewer-publish"]
        else:
            args += ["--no-viewer-publish"]
        if request.show_depth:
            args += ["--show-depth"]
        if not request.memory_store:
            args += ["--no-memory-store"]
        if not request.detection_enabled:
            args += ["--skip-detection"]
        if request.planner_mode == "pointgoal":
            assert request.goal_x is not None and request.goal_y is not None
            args += ["--goal-x", str(request.goal_x), "--goal-y", str(request.goal_y)]
        return args

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
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
            wait_fn = getattr(process, "wait", None)
            if callable(wait_fn):
                try:
                    await asyncio.to_thread(wait_fn, 5.0)
                except Exception:
                    kill = getattr(process, "kill", None)
                    if callable(kill):
                        kill()
        if managed.stdout_handle is not None:
            managed.stdout_handle.close()
        if managed.stderr_handle is not None:
            managed.stderr_handle.close()

    @staticmethod
    def _tcp_ready(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            return False

    def _default_health_url(self, name: str) -> str:
        if name == "navdp":
            return "http://127.0.0.1:8888/health"
        if name == "dual":
            return "http://127.0.0.1:8890/health"
        if name == "system2":
            return "http://127.0.0.1:8080"
        return ""

    def _default_runner(self, spec: ProcessSpec, stdout_log: Path, stderr_log: Path) -> ManagedProcess:
        stdout_handle = stdout_log.open("wb")
        stderr_handle = stderr_log.open("wb")
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
