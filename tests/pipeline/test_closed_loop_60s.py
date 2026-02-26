#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.pipeline.pipeline_test_utils import (
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
    phase = "closed_loop_60s"
    run_dir = init_run_dir(phase=phase, run_dir=args.run_dir)
    manip_cfg = build_manip_cfg(config_path=Path(args.config), run_dir=run_dir, phase=phase)
    manip = GrootManipulator(manip_cfg)
    err: str | None = None
    try:
        await manip.warmup()
        if not bool(getattr(manip, "locomotion_enabled", False)):
            raise RuntimeError("No locomotion backend is enabled after warmup.")
        manip.set_telemetry_phase(phase)

        await manip.execute_locomotion(
            linear_x=0.2,
            linear_y=0.0,
            angular_z=0.0,
            duration_s=20.0,
            source="test_closed_loop_60s",
        )
        await manip.execute_locomotion(
            linear_x=0.3,
            linear_y=0.0,
            angular_z=0.0,
            duration_s=20.0,
            source="test_closed_loop_60s",
        )
        for idx in range(20):
            yaw = 0.3 if (idx % 2 == 0) else -0.3
            await manip.execute_locomotion(
                linear_x=0.2,
                linear_y=0.0,
                angular_z=yaw,
                duration_s=1.0,
                source="test_closed_loop_60s",
            )
    except Exception as exc:
        err = str(exc)
        logging.exception("Closed-loop test failed")
    finally:
        await manip.stop()

    records = load_phase_records(run_dir, phase)
    metrics = summarize_phase_metrics(records)
    closed_pass = bool(
        err is None
        and not bool(metrics.get("fall_flag"))
        and not bool(metrics.get("locomotion_req_all_zero"))
        and metrics.get("max_abs_action") is not None
        and float(metrics.get("max_abs_action") or 0.0) <= 2.5
    )

    mapping = load_joint_map_summary(DEFAULT_JOINT_MAP)
    action_scale = action_scale_and_defaults(records)
    standing = {"pass": "n/a", "max_abs_action": "n/a", "max_roll": "n/a", "max_pitch": "n/a"}
    walking = {
        "completed": "n/a",
        "aborted": "n/a",
        "abort_step": "n/a",
        "abort_vx": "n/a",
        "actions_at_abort": None,
        "last_1s_stats": "n/a",
    }
    closed_loop = {
        "pass": closed_pass,
        "max_roll": metrics.get("max_roll"),
        "max_pitch": metrics.get("max_pitch"),
        "min_base_height": metrics.get("min_base_height"),
        "tracking_err_rms": metrics.get("tracking_err_rms"),
        "server_latency_p95": metrics.get("server_latency_p95"),
        "loop_jitter_std": metrics.get("loop_jitter_std"),
        "overrun_count": metrics.get("overrun_count"),
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
    print(f"result: {'PASS' if closed_pass else 'FAIL'}")
    print(
        "summary:",
        {
            "max_roll": metrics.get("max_roll"),
            "max_pitch": metrics.get("max_pitch"),
            "min_base_height": metrics.get("min_base_height"),
            "tracking_err_rms": metrics.get("tracking_err_rms"),
            "server_latency_p95": metrics.get("server_latency_p95"),
            "loop_jitter_std": metrics.get("loop_jitter_std"),
            "overrun_count": metrics.get("overrun_count"),
        },
    )
    return 0 if closed_pass else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="60-second closed-loop pipeline stability test.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    rc = asyncio.run(_run(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
