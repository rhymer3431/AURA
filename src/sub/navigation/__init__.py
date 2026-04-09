from __future__ import annotations

from pathlib import Path
from typing import Final

from ..shared import load_module, root_path


NAME: Final[str] = "Navigation Subsystem"
DESCRIPTION: Final[str] = "Point-goal navigation runtime, NavDP client/server boundary, and path following."

MODULES: Final[dict[str, str]] = {
    "runtime": "g1_play.navdp_runtime",
    "client": "g1_play.navdp_client",
    "follower": "g1_play.navdp_follower",
    "geometry": "g1_play.navdp_geometry",
    "goals": "g1_play.navdp_goals",
    "server": "navdp.navdp_server",
    "policy_agent": "navdp.policy_agent",
}

ENTRYPOINTS: Final[dict[str, Path]] = {
    "runtime_launcher": root_path("play_g1_internvla_navdp.py"),
    "navdp_server_launcher": root_path("run_navdp_server_windows.bat"),
}

PUBLIC_APIS: Final[tuple[str, ...]] = (
    "g1_play.navdp_runtime.InternVlaNavDpController",
    "g1_play.navdp_runtime.NavDpPointGoalController",
    "g1_play.navdp_client.NavDpClient",
    "g1_play.navdp_follower.HolonomicPurePursuitFollower",
    "g1_play.navdp_goals.PointGoalProvider",
    "navdp.navdp_server",
)


def load(alias: str):
    return load_module(MODULES[alias])
