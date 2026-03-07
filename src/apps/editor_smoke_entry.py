from __future__ import annotations

import argparse

from apps.live_smoke_app import parse_args
from runtime.isaac_launch_modes import LAUNCH_MODE_EDITOR_ASSISTED, LAUNCH_MODE_EXTENSION
from runtime.live_smoke_runner import LiveSmokeRunner


def build_editor_smoke_args(
    argv: list[str] | None = None,
    *,
    launch_mode: str = LAUNCH_MODE_EDITOR_ASSISTED,
) -> argparse.Namespace:
    args = parse_args(argv or [])
    if "--mode" not in list(getattr(args, "_argv", [])):
        args.mode = "smoke"
    if "--launch-mode" not in list(getattr(args, "_argv", [])):
        args.launch_mode = str(launch_mode)
    return args


def run_editor_smoke(
    argv: list[str] | None = None,
    *,
    simulation_app=None,
    stage=None,
    launch_mode: str = LAUNCH_MODE_EDITOR_ASSISTED,
) -> int:
    args = build_editor_smoke_args(argv, launch_mode=launch_mode)
    normalized_launch_mode = str(args.launch_mode).strip().lower() or str(launch_mode)
    if normalized_launch_mode not in {LAUNCH_MODE_EDITOR_ASSISTED, LAUNCH_MODE_EXTENSION}:
        normalized_launch_mode = str(launch_mode)
        args.launch_mode = normalized_launch_mode
    runner = LiveSmokeRunner(args)
    return runner.run_in_editor(
        simulation_app=simulation_app,
        stage=stage,
        launch_mode=normalized_launch_mode,
    )


def run_extension_smoke(argv: list[str] | None = None, *, simulation_app=None, stage=None) -> int:
    return run_editor_smoke(
        argv=argv,
        simulation_app=simulation_app,
        stage=stage,
        launch_mode=LAUNCH_MODE_EXTENSION,
    )


def main(argv: list[str] | None = None) -> int:
    return run_editor_smoke(argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
