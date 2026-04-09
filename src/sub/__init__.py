from __future__ import annotations

from . import control, inference, navigation, planner, world_state


SUBSYSTEMS = {
    "navigation": navigation,
    "inference": inference,
    "world_state": world_state,
    "planner": planner,
    "control": control,
}


def subsystem_names() -> tuple[str, ...]:
    return tuple(SUBSYSTEMS.keys())
