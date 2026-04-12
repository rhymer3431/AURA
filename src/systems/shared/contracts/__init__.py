"""Common DTOs used across subsystem APIs.

This package is imported very early by navigation and planner modules. Keep the
top-level exports lazy so importing a narrow contract module such as
`systems.shared.contracts.navigation` does not pull in world-state runtime
types and re-enter navigation during package initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .dashboard import LogRecord, ProcessRecord, ServiceSnapshot
from .inference import System2Result
from .navigation import NavDpPlan
from .observation import (
    HistoryView,
    LocomotionCommand,
    NavigationSessionSpec,
    ObservationFrame,
    PlannerInput,
    RawObservation,
    TrajectoryPlan,
    decode_rgb_history_npz,
    encode_rgb_history_npz,
)
from .planner import Subgoal, TaskFrame

_RUNTIME_STATE_EXPORTS = {
    "ActionOverrideState",
    "CaptureState",
    "CommandState",
    "HistoryView",
    "GoalState",
    "LocomotionState",
    "LocomotionCommand",
    "NavDpState",
    "NavigationSessionSpec",
    "NavigationPipelineState",
    "ObservationFrame",
    "PlannerInput",
    "RawObservation",
    "StatusState",
    "System2RuntimeState",
    "TaskExecutionState",
    "TrajectoryPlan",
}


def __getattr__(name: str) -> Any:
    if name in _RUNTIME_STATE_EXPORTS:
        module = import_module(".runtime_state", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionOverrideState",
    "CaptureState",
    "CommandState",
    "decode_rgb_history_npz",
    "encode_rgb_history_npz",
    "GoalState",
    "HistoryView",
    "LogRecord",
    "LocomotionState",
    "LocomotionCommand",
    "NavDpPlan",
    "NavDpState",
    "NavigationSessionSpec",
    "NavigationPipelineState",
    "ObservationFrame",
    "ProcessRecord",
    "PlannerInput",
    "RawObservation",
    "ServiceSnapshot",
    "StatusState",
    "Subgoal",
    "System2Result",
    "System2RuntimeState",
    "TaskExecutionState",
    "TaskFrame",
    "TrajectoryPlan",
]
