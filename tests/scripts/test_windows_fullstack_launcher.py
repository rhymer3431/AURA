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
        "AURA_PYTHON",
        "INFERENCE_SYSTEM_HOST",
        "INFERENCE_SYSTEM_PORT",
        "NAVDP_HOST",
        "NAVDP_PORT",
        "SYSTEM2_HOST",
        "SYSTEM2_PORT",
        "PLANNER_MODEL_HOST",
        "PLANNER_MODEL_PORT",
        "PLANNER_SYSTEM_HOST",
        "PLANNER_SYSTEM_PORT",
        "NAVIGATION_SYSTEM_HOST",
        "NAVIGATION_SYSTEM_PORT",
        "ISAACSIM_PATH",
        "RUNTIME_CONTROL_API_HOST",
        "RUNTIME_CONTROL_API_PORT",
        "NAVIGATION_URL",
        "NAVDP_URL",
        "SYSTEM2_URL",
        "NAVIGATION_SYSTEM_URL",
        "NAVIGATION_NAVDP_FALLBACK",
        "PLANNER_MODEL_BASE_URL",
        "AURA_LAUNCH_MODE",
        "AURA_VIEWER_ENABLED",
        "AURA_MEMORY_STORE",
        "AURA_DETECTION_ENABLED",
        "AURA_RUNTIME_URL",
        "AURA_RUNTIME_SUPERVISOR_URL",
        "AURA_INFERENCE_SYSTEM_URL",
        "AURA_PLANNER_SYSTEM_URL",
        "AURA_NAVIGATION_SYSTEM_URL",
        "AURA_CONTROL_RUNTIME_URL",
        "AURA_WEBRTC_PROXY_BASE",
        "AURA_WEBRTC_RGB_FPS",
        "AURA_WEBRTC_DEPTH_FPS",
        "AURA_WEBRTC_TELEMETRY_HZ",
        "AURA_WEBRTC_POLL_INTERVAL_MS",
        "AURA_WEBRTC_ENABLE_DEPTH_TRACK",
        "AURA_DASHBOARD_API_BASE_URL",
    ):
        env.pop(key, None)
    return env


def _load_json_from_stdout(stdout: str) -> dict[str, object]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip() != ""]
    assert lines, "launcher stdout was empty"
    return json.loads(lines[-1])


def test_inference_system_launcher_reports_active_contract() -> None:
    launcher = ROOT / "scripts" / "run_system" / "inference_system_windows.bat"
    env = _base_env()
    env.update(
        {
            "INFERENCE_SYSTEM_PORT": "16880",
            "NAVDP_PORT": "18890",
            "SYSTEM2_PORT": "15813",
            "PLANNER_MODEL_PORT": "8095",
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
    assert payload["inference_system_port"] == 16880
    assert payload["navdp_url"] == "http://127.0.0.1:18890"
    assert payload["system2_url"] == "http://127.0.0.1:15813"
    assert payload["planner_model_url"] == "http://127.0.0.1:8095/v1/chat/completions"


def test_navigation_system_launcher_reports_active_contract() -> None:
    launcher = ROOT / "scripts" / "run_system" / "navigation_system_windows.bat"
    env = _base_env()
    env.update(
        {
            "NAVIGATION_SYSTEM_PORT": "17892",
            "SYSTEM2_URL": "http://127.0.0.1:15813",
            "NAVDP_URL": "http://127.0.0.1:18890",
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
    assert payload["navigation_system_port"] == 17892
    assert payload["system2_url"] == "http://127.0.0.1:15813"
    assert payload["navdp_url"] == "http://127.0.0.1:18890"
    assert payload["navdp_fallback"] == "heuristic"


def test_planner_system_launcher_reports_active_contract() -> None:
    launcher = ROOT / "scripts" / "run_system" / "planner_system_windows.bat"
    env = _base_env()
    env.update(
        {
            "PLANNER_SYSTEM_PORT": "17891",
            "NAVIGATION_SYSTEM_URL": "http://127.0.0.1:17892",
            "PLANNER_MODEL_BASE_URL": "http://127.0.0.1:8095/v1/chat/completions",
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
    assert payload["planner_system_port"] == 17891
    assert payload["navigation_system_url"] == "http://127.0.0.1:17892"
    assert payload["planner_model_base_url"] == "http://127.0.0.1:8095/v1/chat/completions"


def test_control_runtime_launcher_reports_active_contract() -> None:
    launcher = ROOT / "scripts" / "run_system" / "control_runtime_windows.bat"
    env = _base_env()
    env.update(
        {
            "RUNTIME_CONTROL_API_PORT": "8898",
            "NAVIGATION_URL": "http://127.0.0.1:17890",
            "AURA_LAUNCH_MODE": "headless",
            "AURA_VIEWER_ENABLED": "0",
            "AURA_MEMORY_STORE": "1",
            "AURA_DETECTION_ENABLED": "0",
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
    assert payload["runtime_control_api_port"] == 8898
    assert payload["navigation_url"] == "http://127.0.0.1:17890"
    assert payload["launch_mode"] == "headless"
    assert payload["viewer_enabled"] is False
    assert payload["viewer_publish"] is False
    assert payload["memory_store"] is True
    assert payload["detection_enabled"] is False


def test_canonical_launchers_do_not_reference_removed_helpers() -> None:
    control_text = (ROOT / "scripts" / "run_system" / "control_runtime_windows.bat").read_text(encoding="utf-8", errors="ignore")
    inference_text = (ROOT / "scripts" / "run_system" / "inference_system_windows.bat").read_text(encoding="utf-8", errors="ignore")
    backend_text = (ROOT / "scripts" / "run_system" / "backend_windows.ps1").read_text(encoding="utf-8", errors="ignore")
    runtime_text = (ROOT / "scripts" / "run_system" / "runtime_windows.ps1").read_text(encoding="utf-8", errors="ignore")

    assert "send_internvla_nav_command" not in control_text
    assert "src\\systems\\navigation\\bin\\run_navdp_server_windows.bat" not in control_text
    assert "src\\systems\\inference\\bin\\run_internvla_nav_server_windows.bat" not in control_text
    assert "serve_planner_qwen3_nothink.ps1" not in control_text
    assert "systems.inference.api.serve_inference_system" in inference_text
    assert "backend.api.serve_backend" in backend_text
    assert "dashboard\\python" not in backend_text
    assert "runtime.api.serve_runtime" in runtime_text


def test_backend_launcher_uses_canonical_environment_names() -> None:
    launcher_text = (ROOT / "scripts" / "run_system" / "backend_windows.ps1").read_text(encoding="utf-8", errors="ignore")

    assert "AURA_RUNTIME_URL" in launcher_text
    assert "AURA_RUNTIME_SUPERVISOR_URL" in launcher_text
    assert "AURA_INFERENCE_SYSTEM_URL" in launcher_text
    assert "AURA_PLANNER_SYSTEM_URL" in launcher_text
    assert "AURA_NAVIGATION_SYSTEM_URL" in launcher_text
    assert "AURA_CONTROL_RUNTIME_URL" in launcher_text
    assert "AURA_WEBRTC_RGB_FPS" in launcher_text
    assert "AURA_WEBRTC_ENABLE_DEPTH_TRACK" in launcher_text


def test_backend_launcher_omits_runtime_flag_when_backend_owns_runtime(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "\n".join(
            [
                "@echo off",
                "echo %*",
                "exit /b 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "backend_windows.ps1"),
            "-Python",
            str(fake_python),
        ],
        cwd=str(ROOT),
        env=_base_env(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "--runtime-url" not in completed.stdout
