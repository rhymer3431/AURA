from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.bootstrap_diagnostics import BootstrapPhaseTracker


def test_bootstrap_phase_tracker_writes_running_and_failure_artifacts(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "diagnostics.json"
    tracker = BootstrapPhaseTracker(
        diagnostics_path=diagnostics_path,
        artifact_dir=tmp_path / "artifacts",
        launch_mode="standalone_python",
        frame_source="live",
        headless=True,
        cli_args=["--headless"],
        phase_timeouts={"simulation_app_created": 30.0},
    )

    tracker.start_phase("simulation_app_created", context={"headless": True})
    running_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    running_phase = next(phase for phase in running_payload["phases"] if phase["name"] == "simulation_app_created")
    assert running_payload["current_phase"] == "simulation_app_created"
    assert running_phase["status"] == "running"
    assert running_phase["timeout_sec"] == 30.0

    tracker.fail_phase("simulation_app_created", message="stalled bootstrap", timeout=True)
    tracker.finalize_failure()
    failed_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    failed_phase = next(phase for phase in failed_payload["phases"] if phase["name"] == "simulation_app_created")
    assert failed_payload["status"] == "failed"
    assert failed_payload["failure_phase"] == "simulation_app_created"
    assert failed_phase["status"] == "timeout"
    assert "stalled bootstrap" in failed_payload["summary"]


def test_bootstrap_phase_tracker_writes_named_artifacts(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "diagnostics.json"
    tracker = BootstrapPhaseTracker(
        diagnostics_path=diagnostics_path,
        artifact_dir=tmp_path / "artifacts",
        launch_mode="standalone_python",
        frame_source="live",
        headless=False,
    )

    tracker.write_json_artifact("cli_args", filename="cli_args.json", payload={"argv": ["--mode", "smoke"]})
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert "cli_args" in payload["artifacts"]
    cli_args_path = Path(payload["artifacts"]["cli_args"])
    assert cli_args_path.exists()
    assert json.loads(cli_args_path.read_text(encoding="utf-8"))["argv"] == ["--mode", "smoke"]
