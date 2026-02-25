#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from pipeline_test_utils import (
    DEFAULT_CONFIG,
    DEFAULT_JOINT_MAP,
    GrootManipulator,
    action_scale_and_defaults,
    build_manip_cfg,
    extract_pd_gain_rows,
    init_run_dir,
    load_joint_map_summary,
    load_phase_records,
    render_report,
    summarize_phase_metrics,
    write_report,
)


async def _run(args: argparse.Namespace) -> int:
    phase = "standing"
    run_dir = init_run_dir(phase=phase, run_dir=args.run_dir)
    manip_cfg = build_manip_cfg(config_path=Path(args.config), run_dir=run_dir, phase=phase)
    manip = GrootManipulator(manip_cfg)
    err: str | None = None
    try:
        await manip.warmup()
        if not manip.sonic_enabled:
            raise RuntimeError("SONIC backend is not enabled after warmup.")
        manip.set_telemetry_phase(phase)
        await manip.execute_locomotion(
            linear_x=0.0,
            linear_y=0.0,
            angular_z=0.0,
            duration_s=5.0,
            source="test_standing_pipeline",
        )
    except Exception as exc:
        err = str(exc)
        logging.exception("Standing test failed")
    finally:
        await manip.stop()

    records = load_phase_records(run_dir, phase)
    metrics = summarize_phase_metrics(records)
    roll_ok = (metrics.get("max_roll") is not None) and (float(metrics.get("max_roll") or 0.0) <= args.max_roll)
    pitch_ok = (metrics.get("max_pitch") is not None) and (float(metrics.get("max_pitch") or 0.0) <= args.max_pitch)
    standing_pass = bool(
        err is None
        and not bool(metrics.get("fall_flag"))
        and metrics.get("max_abs_action") is not None
        and float(metrics.get("max_abs_action") or 0.0) <= 2.5
        and roll_ok
        and pitch_ok
    )

    mapping = load_joint_map_summary(DEFAULT_JOINT_MAP)
    action_scale = action_scale_and_defaults(records)
    standing = {
        "pass": standing_pass,
        "max_abs_action": metrics.get("max_abs_action"),
        "max_roll": metrics.get("max_roll"),
        "max_pitch": metrics.get("max_pitch"),
    }
    walking = {
        "completed": "n/a",
        "aborted": "n/a",
        "abort_step": "n/a",
        "abort_vx": "n/a",
        "actions_at_abort": None,
        "last_1s_stats": "n/a",
    }
    closed_loop = {
        "pass": "n/a",
        "max_roll": "n/a",
        "max_pitch": "n/a",
        "min_base_height": "n/a",
        "tracking_err_rms": "n/a",
        "server_latency_p95": "n/a",
        "loop_jitter_std": "n/a",
        "overrun_count": "n/a",
    }
    pd_rows = extract_pd_gain_rows(records)
    report = render_report(
        run_dir=run_dir,
        phase=phase,
        mapping=mapping,
        action_scale=action_scale,
        control_metrics=metrics,
        standing=standing,
        walking=walking,
        closed_loop=closed_loop,
        pd_gain_rows=pd_rows,
    )
    report_path = write_report(run_dir, report)
    print(f"run_dir: {run_dir}")
    print(f"report: {report_path}")
    if err is not None:
        print(f"result: FAIL ({err})")
        return 1
    print(f"result: {'PASS' if standing_pass else 'FAIL'}")
    print(f"max_abs_action: {metrics.get('max_abs_action')}")
    print(f"max_roll: {metrics.get('max_roll')} max_pitch: {metrics.get('max_pitch')}")
    return 0 if standing_pass else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-second standing pipeline test with runtime telemetry.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--max-roll", type=float, default=0.60)
    parser.add_argument("--max-pitch", type=float, default=0.60)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    rc = asyncio.run(_run(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
