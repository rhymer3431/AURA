from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mission.mission_manager import MissionManager
from planning.coordinator import PlanningCoordinator
from runtime.aura_runtime import AuraRuntimeCommandSource, NavigationRuntime
from services.dual_orchestrator import DualOrchestrator
from services.task_orchestrator import TaskOrchestrator


def test_navigation_runtime_legacy_alias_remains_available() -> None:
    assert issubclass(AuraRuntimeCommandSource, NavigationRuntime)


def test_mission_manager_wraps_legacy_task_orchestrator() -> None:
    assert issubclass(MissionManager, TaskOrchestrator)


def test_planning_coordinator_wraps_legacy_dual_orchestrator() -> None:
    assert issubclass(PlanningCoordinator, DualOrchestrator)
