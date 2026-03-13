from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard_backend.config import DashboardBackendConfig
from dashboard_backend.models import parse_session_request
from dashboard_backend.process_manager import ManagedProcess, ProcessManager


class _FakeProcess:
    def __init__(self, name: str, events: list[tuple[str, str]]) -> None:
        self.name = name
        self.pid = hash(name) & 0xFFFF
        self.returncode = None
        self._events = events

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self._events.append(("terminate", self.name))
        self.returncode = 0

    def wait(self, _timeout: float | None = None) -> int:
        self._events.append(("wait", self.name))
        self.returncode = 0
        return 0


def test_process_manager_starts_interactive_stack_and_stops_in_reverse_order(monkeypatch: pytest.MonkeyPatch) -> None:
    created_specs = []
    lifecycle_events: list[tuple[str, str]] = []

    def runner(spec, stdout_log: Path, stderr_log: Path) -> ManagedProcess:  # noqa: ANN001
        created_specs.append(spec)
        return ManagedProcess(
            spec=spec,
            process=_FakeProcess(spec.name, lifecycle_events),
            started_at=time.time(),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

    async def no_wait(*_args, **_kwargs) -> None:
        return None

    config = DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard")
    manager = ProcessManager(config, runner=runner)
    monkeypatch.setattr(manager, "_wait_ready", no_wait)
    monkeypatch.setattr(manager, "_reserve_port", lambda _host, preferred_port, reserved=None: preferred_port)

    request = parse_session_request(
        {
            "plannerMode": "interactive",
            "launchMode": "headless",
            "scenePreset": "warehouse",
            "viewerEnabled": True,
            "showDepth": True,
            "memoryStore": False,
            "detectionEnabled": False,
        }
    )

    asyncio.run(manager.start_session(request))
    snapshot = manager.snapshot()

    assert [spec.name for spec in created_specs] == ["navdp", "system2", "dual", "runtime"]
    assert created_specs[-1].args == (
        "--planner-mode",
        "interactive",
        "--scene-preset",
        "warehouse",
        "--native-viewer",
        "off",
        "--server-url",
        "http://127.0.0.1:8888",
        "--dual-server-url",
        "http://127.0.0.1:8890",
        "--headless",
        "--viewer-publish",
        "--show-depth",
        "--no-memory-store",
        "--skip-detection",
    )
    assert [item["name"] for item in snapshot if item["state"] == "running"] == ["navdp", "system2", "dual", "runtime"]

    asyncio.run(manager.stop_all())

    assert [event for event in lifecycle_events if event[0] == "terminate"] == [
        ("terminate", "runtime"),
        ("terminate", "dual"),
        ("terminate", "system2"),
        ("terminate", "navdp"),
    ]


def test_process_manager_marks_optional_services_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def runner(spec, stdout_log: Path, stderr_log: Path) -> ManagedProcess:  # noqa: ANN001
        return ManagedProcess(
            spec=spec,
            process=_FakeProcess(spec.name, []),
            started_at=time.time(),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

    async def no_wait(*_args, **_kwargs) -> None:
        return None

    config = DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard")
    manager = ProcessManager(config, runner=runner)
    monkeypatch.setattr(manager, "_wait_ready", no_wait)
    monkeypatch.setattr(manager, "_reserve_port", lambda _host, preferred_port, reserved=None: preferred_port)

    request = parse_session_request(
        {
            "plannerMode": "pointgoal",
            "launchMode": "gui",
            "scenePreset": "warehouse",
            "viewerEnabled": False,
            "showDepth": False,
            "memoryStore": True,
            "detectionEnabled": True,
            "goal": {"x": 2.0, "y": 0.0},
        }
    )

    asyncio.run(manager.start_session(request))
    snapshot = {item["name"]: item for item in manager.snapshot()}

    assert snapshot["navdp"]["state"] == "running"
    assert snapshot["runtime"]["state"] == "running"
    assert snapshot["system2"]["state"] == "not_required"
    assert snapshot["dual"]["state"] == "not_required"


def test_process_manager_propagates_allocated_service_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    created_specs = []

    def runner(spec, stdout_log: Path, stderr_log: Path) -> ManagedProcess:  # noqa: ANN001
        created_specs.append(spec)
        return ManagedProcess(
            spec=spec,
            process=_FakeProcess(spec.name, []),
            started_at=time.time(),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

    async def no_wait(*_args, **_kwargs) -> None:
        return None

    config = DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard")
    manager = ProcessManager(config, runner=runner)
    monkeypatch.setattr(manager, "_wait_ready", no_wait)
    allocated_ports = iter([18888, 18080, 18890])
    monkeypatch.setattr(manager, "_reserve_port", lambda _host, _preferred_port, reserved=None: next(allocated_ports))

    request = parse_session_request(
        {
            "plannerMode": "interactive",
            "launchMode": "headless",
            "scenePreset": "warehouse",
            "viewerEnabled": True,
            "showDepth": False,
            "memoryStore": True,
            "detectionEnabled": True,
        }
    )

    asyncio.run(manager.start_session(request))

    navdp_spec = next(spec for spec in created_specs if spec.name == "navdp")
    system2_spec = next(spec for spec in created_specs if spec.name == "system2")
    dual_spec = next(spec for spec in created_specs if spec.name == "dual")
    runtime_spec = next(spec for spec in created_specs if spec.name == "runtime")

    assert navdp_spec.health_url == "http://127.0.0.1:18888/health"
    assert dict(navdp_spec.env)["NAVDP_PORT"] == "18888"
    assert system2_spec.health_url == "http://127.0.0.1:18080"
    assert dict(system2_spec.env)["INTERNVLA_PORT"] == "18080"
    assert dual_spec.health_url == "http://127.0.0.1:18890/health"
    assert dict(dual_spec.env)["DUAL_NAVDP_URL"] == "http://127.0.0.1:18888"
    assert dict(dual_spec.env)["DUAL_VLM_URL"] == "http://127.0.0.1:18080"
    assert "--server-url" in runtime_spec.args
    assert "http://127.0.0.1:18888" in runtime_spec.args
    assert "--dual-server-url" in runtime_spec.args
    assert "http://127.0.0.1:18890" in runtime_spec.args


def test_process_manager_uses_windows_process_tree_kill_for_running_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    lifecycle_events: list[tuple[str, str]] = []

    def runner(spec, stdout_log: Path, stderr_log: Path) -> ManagedProcess:  # noqa: ANN001
        return ManagedProcess(
            spec=spec,
            process=_FakeProcess(spec.name, lifecycle_events),
            started_at=time.time(),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

    async def no_wait(*_args, **_kwargs) -> None:
        return None

    config = DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard")
    manager = ProcessManager(config, runner=runner)
    monkeypatch.setattr(manager, "_wait_ready", no_wait)
    monkeypatch.setattr(manager, "_reserve_port", lambda _host, preferred_port, reserved=None: preferred_port)
    monkeypatch.setattr(manager, "_should_kill_process_tree", lambda: True)

    killed_pids: list[int] = []

    def fake_kill_process_tree(pid: int) -> bool:
        killed_pids.append(pid)
        return True

    monkeypatch.setattr(manager, "_kill_process_tree", fake_kill_process_tree)

    request = parse_session_request(
        {
            "plannerMode": "interactive",
            "launchMode": "headless",
            "scenePreset": "warehouse",
            "viewerEnabled": True,
            "showDepth": False,
            "memoryStore": True,
            "detectionEnabled": True,
        }
    )

    asyncio.run(manager.start_session(request))
    processes = [manager._processes[name] for name in ("runtime", "dual", "system2", "navdp")]  # noqa: SLF001
    expected_pids = [managed.pid() for managed in processes]

    asyncio.run(manager.stop_all())

    assert killed_pids == expected_pids
    assert [event for event in lifecycle_events if event[0] == "terminate"] == []
