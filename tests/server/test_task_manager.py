from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from systems.transport.messages import TaskRequest
from server.task_manager import TaskManager


class _FakePlannerCoordinator:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.execution_mode = "IDLE"

    def activate_idle(self, reason: str) -> None:
        self.events.append(f"activate_idle:{reason}")

    def ensure_navdp_service_ready(self, *, context: str) -> None:
        self.events.append(f"navdp:{context}")

    def ensure_system2_service_ready(self, *, context: str) -> None:
        self.events.append(f"system2:{context}")

    def start_nav_task(self, instruction: str, *, mode: str = "NAV") -> None:
        self.events.append(f"start_nav_task:{mode}:{instruction}")

    def set_execution_mode(self, mode: str) -> None:
        self.execution_mode = str(mode)
        self.events.append(f"set_execution_mode:{mode}")


class _FakeMemoryClient:
    def __init__(self) -> None:
        self.set_calls: list[dict[str, object]] = []
        self.clear_calls: list[dict[str, object]] = []

    def set_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self.set_calls.append(dict(kwargs))

    def clear_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self.clear_calls.append(dict(kwargs))


class _Args:
    goal_tolerance_m = 0.4


def test_task_manager_routes_general_embodied_instruction_to_nav() -> None:
    manager = TaskManager(_Args())
    planner_coordinator = _FakePlannerCoordinator()
    memory_client = _FakeMemoryClient()

    notices = manager.handle_event(
        TaskRequest(command_text="go to the loading dock", task_id="task-1"),
        planner_coordinator=planner_coordinator,
        memory_client=memory_client,
    )

    assert manager.mode == "NAV"
    assert manager.snapshot().instruction == "go to the loading dock"
    assert manager.snapshot().state == "active"
    assert "navdp:NAV task (dashboard)" in planner_coordinator.events
    assert "system2:NAV task (dashboard)" in planner_coordinator.events
    assert any(item.startswith("start_nav_task:NAV:go to the loading dock") for item in planner_coordinator.events)
    assert memory_client.set_calls == [
        {
            "instruction": "go to the loading dock",
            "planner_mode": "nav",
            "task_state": "active",
            "task_id": "task-1",
            "command_id": -1,
        }
    ]
    assert notices[0].details["executionMode"] == "NAV"
