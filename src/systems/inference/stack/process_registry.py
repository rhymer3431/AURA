"""Child-process registry for managed inference services."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time

from systems.inference.stack.config import ManagedServiceConfig


CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


@dataclass(slots=True)
class ManagedProcess:
    config: ManagedServiceConfig
    process: subprocess.Popen[str]
    stdout_log: Path
    stderr_log: Path
    started_at: float

    def snapshot(self) -> dict[str, object]:
        exit_code = self.process.poll()
        return {
            "name": self.config.name,
            "pid": self.process.pid,
            "state": "running" if exit_code is None else "exited",
            "exit_code": exit_code,
            "required": self.config.required,
            "started_at": self.started_at,
            "health_url": self.config.health_url,
            "stdout_log": str(self.stdout_log),
            "stderr_log": str(self.stderr_log),
            "command": list(self.config.command),
        }


class ProcessRegistry:
    def __init__(self, log_dir: Path):
        self._log_dir = Path(log_dir)
        self._processes: dict[str, ManagedProcess] = {}

    def start(self, config: ManagedServiceConfig) -> ManagedProcess:
        if config.name in self._processes and self._processes[config.name].process.poll() is None:
            return self._processes[config.name]
        self._log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = self._log_dir / f"{config.name}.stdout.log"
        stderr_log = self._log_dir / f"{config.name}.stderr.log"
        stdout_handle = open(stdout_log, "a", encoding="utf-8")
        stderr_handle = open(stderr_log, "a", encoding="utf-8")
        process = subprocess.Popen(
            config.command,
            cwd=str(Path(__file__).resolve().parents[4]),
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            text=True,
            creationflags=CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        managed = ManagedProcess(
            config=config,
            process=process,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            started_at=time.time(),
        )
        self._processes[config.name] = managed
        return managed

    def stop(self, name: str, *, timeout_s: float = 5.0) -> None:
        managed = self._processes.get(name)
        if managed is None:
            return
        if managed.process.poll() is None:
            managed.process.terminate()
            try:
                managed.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                managed.process.kill()
                managed.process.wait(timeout=timeout_s)

    def stop_all(self) -> None:
        for name in list(self._processes.keys())[::-1]:
            self.stop(name)

    def snapshot(self) -> list[dict[str, object]]:
        return [self._processes[name].snapshot() for name in sorted(self._processes.keys())]
