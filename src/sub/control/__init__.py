from __future__ import annotations

from pathlib import Path
from typing import Final

from ..shared import load_module, root_path


NAME: Final[str] = "Control Subsystem"
DESCRIPTION: Final[str] = "Top-level runtime orchestration, locomotion policy control, command ingestion, and launch surfaces."

MODULES: Final[dict[str, str]] = {
    "args": "g1_play.args",
    "entrypoint": "g1_play.entrypoint",
    "runtime": "g1_play.runtime",
    "controller": "g1_play.controller",
    "command": "g1_play.command",
    "policy_session": "g1_play.policy_session",
    "training_config": "g1_play.training_config",
}

ENTRYPOINTS: Final[dict[str, Path]] = {
    "play_script": root_path("play_g1_internvla_navdp.py"),
    "run_script": root_path("run_sim_g1_internvla_navdp_windows.bat"),
    "command_helper": root_path("send_internvla_nav_command_windows.bat"),
}

PUBLIC_APIS: Final[tuple[str, ...]] = (
    "g1_play.args.build_arg_parser",
    "g1_play.runtime.run",
    "g1_play.controller.G1PolicyController",
    "g1_play.command.ConsoleCmdVelController",
    "g1_play.command.KeyboardCmdVelController",
)


def load(alias: str):
    return load_module(MODULES[alias])
