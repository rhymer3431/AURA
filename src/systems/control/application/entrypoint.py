"""Entrypoint that preserves Isaac Sim bootstrap ordering."""

from __future__ import annotations

from pathlib import Path
import traceback

from systems.control.api.runtime_args import BOOTSTRAP_ARGS, build_arg_parser
from systems.world_state.api.paths import repo_dir


def main():
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": BOOTSTRAP_ARGS.headless})
    try:
        args = build_arg_parser().parse_args()

        from systems.control.application.runtime_orchestrator import run

        run(args, simulation_app)
    except Exception:  # noqa: BLE001
        log_path = Path(repo_dir()) / "artifacts" / "runtime_exception.log"
        error_text = traceback.format_exc()
        print(error_text)
        log_path.write_text(error_text, encoding="utf-8")
        print(f"[ERROR] Runtime exception log written to: {log_path}")
        raise
    finally:
        simulation_app.close()
