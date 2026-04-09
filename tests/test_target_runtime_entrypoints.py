from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from g1_play.args import build_arg_parser


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_arg_parser_exposes_transplanted_flags() -> None:
    args = build_arg_parser().parse_args([])

    assert args.control_mode == "cmd_vel"
    assert hasattr(args, "planner_base_url")
    assert hasattr(args, "planner_model")
    assert hasattr(args, "planner_timeout")
    assert hasattr(args, "nav_command_api_host")
    assert hasattr(args, "nav_command_api_port")
    assert hasattr(args, "camera_api_host")
    assert hasattr(args, "camera_api_port")


def test_root_entrypoints_import_without_target_layout() -> None:
    play_module = _load_module("play_g1_internvla_navdp_smoke", REPO_ROOT / "play_g1_internvla_navdp.py")
    server_module = _load_module("serve_internvla_nav_server_smoke", REPO_ROOT / "serve_internvla_nav_server.py")

    assert callable(play_module.main)
    parsed = server_module.parse_args(["--host", "127.0.0.1", "--port", "15801"])
    assert parsed.host == "127.0.0.1"
    assert parsed.port == 15801
