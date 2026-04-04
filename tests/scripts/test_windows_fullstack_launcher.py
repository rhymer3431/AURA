from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher tests require Windows")

ROOT = Path(__file__).resolve().parents[2]


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "CONDA_ENV_NAME",
        "AURA_CONDA_ENV",
        "CONDA_BAT",
        "CONDA_EXE",
        "AURA_CONDA_EXE",
        "ISAACSIM_PATH",
        "NAVDP_URL",
        "INTERNVLA_URL",
        "G1_POINTGOAL_SERVER_URL",
        "G1_POINTGOAL_SYSTEM2_URL",
        "NAV_INSTRUCTION",
        "NAV_INSTRUCTION_LANGUAGE",
        "NAV_COMMAND_API_HOST",
        "NAV_COMMAND_API_PORT",
        "CAMERA_API_HOST",
        "CAMERA_API_PORT",
        "CAMERA_PITCH_DEG",
    ):
        env.pop(key, None)
    return env


def _load_json_from_stdout(stdout: str) -> dict[str, object]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip() != ""]
    assert lines, "launcher stdout was empty"
    return json.loads(lines[-1])


def test_run_windows_fullstack_helper_prefers_source_env_surface() -> None:
    helper = ROOT / "scripts" / "run_windows_fullstack.ps1"
    env = _base_env()
    env.update(
        {
            "CONDA_ENV_NAME": "source-env",
            "AURA_CONDA_ENV": "compat-env",
            "NAVDP_URL": "http://127.0.0.1:18888",
            "G1_POINTGOAL_SERVER_URL": "http://127.0.0.1:28888",
            "INTERNVLA_URL": "http://127.0.0.1:15811",
            "G1_POINTGOAL_SYSTEM2_URL": "http://127.0.0.1:25811",
            "NAV_INSTRUCTION": "go to source dock",
            "NAV_INSTRUCTION_LANGUAGE": "en",
            "NAV_COMMAND_API_PORT": "9900",
            "CAMERA_API_PORT": "9901",
        }
    )

    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(helper), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["conda_env_name"] == "source-env"
    assert payload["navdp_url"] == "http://127.0.0.1:18888"
    assert payload["internvla_url"] == "http://127.0.0.1:15811"
    assert payload["nav_instruction"] == "go to source dock"
    assert payload["nav_instruction_language"] == "en"
    assert payload["nav_command_api_port"] == 9900
    assert payload["camera_api_port"] == 9901


def test_run_system_all_delegates_to_fullstack_helper() -> None:
    script = ROOT / "scripts" / "run_system.ps1"
    env = _base_env()
    env.update(
        {
            "CONDA_ENV_NAME": "delegate-env",
            "NAVDP_URL": "http://127.0.0.1:18889",
            "INTERNVLA_URL": "http://127.0.0.1:15812",
            "NAV_INSTRUCTION": "delegate instruction",
        }
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Component",
            "all",
            "-PrintConfigJson",
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["conda_env_name"] == "delegate-env"
    assert payload["navdp_url"] == "http://127.0.0.1:18889"
    assert payload["internvla_url"] == "http://127.0.0.1:15812"
    assert payload["nav_instruction"] == "delegate instruction"


def test_batch_launcher_uses_same_contract() -> None:
    launcher = ROOT / "run_sim_g1_internvla_navdp_windows.bat"
    env = _base_env()
    env.update(
        {
            "CONDA_ENV_NAME": "batch-env",
            "NAVDP_URL": "http://127.0.0.1:18890",
            "INTERNVLA_URL": "http://127.0.0.1:15813",
            "NAV_INSTRUCTION": "batch instruction",
        }
    )

    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["conda_env_name"] == "batch-env"
    assert payload["navdp_url"] == "http://127.0.0.1:18890"
    assert payload["internvla_url"] == "http://127.0.0.1:15813"
    assert payload["nav_instruction"] == "batch instruction"
