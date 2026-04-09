from __future__ import annotations

from pathlib import Path
from typing import Final

from ..shared import load_module, root_path


NAME: Final[str] = "Inference Subsystem"
DESCRIPTION: Final[str] = "InternVLA / llama.cpp inference server and multimodal grounding client helpers."

MODULES: Final[dict[str, str]] = {
    "client": "g1_play.internvla_nav",
    "planner_http": "g1_play.tasking.llm_client",
}

ENTRYPOINTS: Final[dict[str, Path]] = {
    "server_script": root_path("serve_internvla_nav_server.py"),
    "server_launcher": root_path("run_internvla_nav_server_windows.bat"),
    "session_checker": root_path("check_internvla_session.py"),
}

PUBLIC_APIS: Final[tuple[str, ...]] = (
    "g1_play.internvla_nav.InternVlaNavClient",
    "g1_play.internvla_nav.System2Result",
    "serve_internvla_nav_server.py:/navigation",
    "serve_internvla_nav_server.py:/healthz",
)


def load(alias: str):
    return load_module(MODULES[alias])
