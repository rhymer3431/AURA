from __future__ import annotations

from pathlib import Path

from sub import SUBSYSTEMS, subsystem_names
from sub import control, inference, navigation, planner, world_state


def test_subsystem_registry_contains_five_requested_domains() -> None:
    assert subsystem_names() == ("navigation", "inference", "world_state", "planner", "control")
    assert set(SUBSYSTEMS) == {"navigation", "inference", "world_state", "planner", "control"}


def test_subsystem_entrypoints_exist() -> None:
    for subsystem in (navigation, inference, world_state, planner, control):
        for path in subsystem.ENTRYPOINTS.values():
            assert Path(path).exists(), path


def test_safe_manifests_can_lazy_load_runtime_modules() -> None:
    assert navigation.load("client").__name__ == "g1_play.navdp_client"
    assert inference.load("client").__name__ == "g1_play.internvla_nav"
    assert planner.load("service").__name__ == "g1_play.tasking.planner_service"
    assert world_state.load("paths").__name__ == "g1_play.paths"
    assert control.load("args").__name__ == "g1_play.args"
