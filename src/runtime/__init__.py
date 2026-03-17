"""Runtime package for NavDP execution flows."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "IsaacRuntime": (".isaac_runtime", "IsaacRuntime"),
    "PlanningSession": (".planning_session", "PlanningSession"),
    "PlannerStats": (".planning_session", "PlannerStats"),
    "Supervisor": (".supervisor", "Supervisor"),
    "SupervisorConfig": (".supervisor", "SupervisorConfig"),
    "TrajectoryUpdate": (".planning_session", "TrajectoryUpdate"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
